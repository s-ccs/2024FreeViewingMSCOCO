using UnfoldBIDS, PyMNE
import BSplineKit
using Unfold
using LazyArtifacts
using DataFrames
using CSV
using Chain
using StatsBase
using CUDA

include("erp_analysis_helper_functions.jl")

data_root_path = "/scratch/data/2024FreeViewingMSCOCO"
layout_df = bids_layout(data_root_path, specific_folder="preprocessed")

data_subset = layout_df[1:2, :]
if data_subset isa DataFrameRow
    data_subset = DataFrame(data_subset)
end

data_df_orig = load_bids_eeg_data(data_subset)
#data_df_orig = load_bids_eeg_data(layout_df)
sfreq = pyconvert(Float64, data_df_orig[1, :raw].info["sfreq"])
channels = pyconvert(Vector, data_df_orig[1, :raw].ch_names)

data_df = deepcopy(data_df_orig)
windows_to_remove = Dict()

for row in eachrow(data_df)
    @assert pyconvert(Float64, row.raw.info["sfreq"]) == sfreq "All recordings should have the same sampling frequency."

    # Shift onset by first_time to match events with EEG data
    first_time = pyconvert(Float64, row.raw.first_time)
    row.events.onset = row.events.onset .- first_time
    row.events.end_time = row.events.end_time .- first_time

    if any(row.events.onset .< 0)
        @warn "For subject $(row[:subject]), there are events with negative onsets. `First_time` from raw: $first_time"
    end

    # Exlcude blink saccades
    row.events = filter(r->r.blink_saccade == false, row.events)

    # Add latency column which is the onset converted to samples
    row.events.latency = row.events.onset * sfreq

    # Only use the data of the specified eye
    row.events = filter(r -> (ismissing(r.eye)) || r.eye == "R", row.events)

    # # Add the saccade amplitude of the previous saccade to each fixation
    # copy_eventinfo!(row.events, "saccade" => "fixation", :sacc_visual_angle; search_fun=:forward, column="trial_type")

    # Exlude practice trials (block 0)
    row.events = filter(r -> (ismissing(r.block) || r.block .!= 0), row.events)

    # TODO: do we need a buffer before stimulus start and after stimulus end?
    stimulus_start = filter(r -> r.trial_type == "02 Stimulus image shown", row.events).latency
    stimulus_end = filter(r -> r.trial_type == "08 Stimulus end", row.events).latency
    within_timewindow = [any(stimulus_start .<= lat .<= stimulus_end) for lat in row.events.latency]
    row.events = row.events[within_timewindow, :]

    if any(row.events.onset .< 0)
        @warn "Even after filtering for events within/around trials, there are negative onsets."
    end

    # Create new columns in the events df for winsorised predictor values
    prop = 0.01
    saccades_view = filter(r->r.trial_type=="saccade", row.events, view=true)
    saccades_view[:, :sacc_visual_angle_w] .= winsor(saccades_view.sacc_visual_angle, prop=prop)
    saccades_view[:, :sacc_duration_w] .= winsor(saccades_view.duration, prop=prop)

    idx_remove = []

    # Remove fixations outside of the screen
    append!(idx_remove, findall(r -> r.trial_type == "fixation" && outside_screen(r.fix_avg_x, r.fix_avg_y), eachrow(row.events)))

    # Remove saccades that start or land outside of the screen
    append!(idx_remove, findall(r -> r.trial_type == "saccade" && (outside_screen(r.sacc_start_x, r.sacc_start_y) || outside_screen(r.sacc_end_x, r.sacc_end_y)), eachrow(row.events)))

    # Save onset and end_time of the events that will be removed in a dictionary
    windows_to_remove[row[:subject]] = row.events[idx_remove, ["onset", "end_time"]]
    # TODO: Decide whether to remove events outside of the screen or not and adapt code accordingly
    # windows_to_remove[row[:subject]] = DataFrame()

    # Remove events from the dataframe
    row.events = row.events[Not(idx_remove), :]

    # Change trial_type for eye-movement events outside of the screen
    # row.events[idx_remove, :trial_type] = row.events[idx_remove, :trial_type] .* "_outside_screen"
    @info "For subject $(row[:subject]), found $(length(idx_remove)) fixations and saccades that were outside of the screen."

    # events_to_remove = findall(r -> r.trial_type == "fixation" && ismissing(r.sacc_visual_angle), eachrow(row.events))
    # row.events = row.events[Not(events_to_remove), :]

    subject_path = construct_subject_bids_path(data_root_path, string(row[:subject]), "erp-analysis")
    mkpath(subject_path)
    file_name = construct_subject_filename(row[:subject], "events", "tsv", session=row.ses, run=row.run, task=row.task)

    CSV.write(joinpath(subject_path, file_name), row.events, delim="\t")
end

function prepare_eeg_data(raw; windows_to_remove_all=Dict(), channels::AbstractVector{<:Union{String,Integer}}=[])

    # Load EEG data
    data = pyconvert(Array, raw.copy().get_data(picks=pylist(channels), units="uV"))

    # Get subject id from raw object
    subject_id = split(pyconvert(String, raw.info["subject_info"]["his_id"]), "sub-")[2]

    # Extract time windows that should be removed for the specific subject
    windows_to_remove_sub = windows_to_remove_all[subject_id]

    if ~isempty(windows_to_remove_sub)
        # Get start and end of data windows that should be removed in samples
        windows_to_remove = @chain windows_to_remove_sub begin
            transform(:onset => (t -> Int.(round.(t*sfreq))) => :start_latency)
            transform(:end_time => (t -> Int.(round.(t*sfreq))) => :end_latency)
        end

        data_to_remove = [windows_to_remove.start_latency windows_to_remove.end_latency]

        cleaned_data = Unfold.clean_data(data, data_to_remove)

    else
        # cleaned_data = data
        # TODO: Change later; Circumvent UnfoldBIDS bug
        cleaned_data = Array{Union{Float64,Missing}}(data)
        #cleaned_data = data[end, end] = missing
    end

    # Define as CUDA array for GPU fitting
    #cleaned_data_cuda = cu(cleaned_data) # TODO: Change once GPU fitting works again in Unfold
    #println(typeof(cleaned_data_cuda))

    return cleaned_data
end

# bf_stimulus = firbasis(τ=(-0.3, 1.5), sfreq=sfreq)
# bf_fixation = firbasis(τ=(-0.3, 1.0), sfreq=sfreq)
# f_stimulus = @formula 0 ~ 1
# f_fixation = @formula 0 ~ 1 + spl(sacc_visual_angle, 4)
# bfDict = ["02 Stimulus image shown" => (f_stimulus, bf_stimulus),
#     "fixation" => (f_fixation, bf_fixation)]
# results_all = run_unfold(data_df, bfDict; eventcolumn="trial_type", channels=["eeg"])

# Model fitting
bf_stimulus = firbasis(τ=(-0.3, 1.5), sfreq=sfreq)
bf_saccade = firbasis(τ=(-0.3, 1.0), sfreq=sfreq)
# bf_saccade_outside = firbasis(τ=(-0.3, 1.0), sfreq=sfreq)

f_stimulus = @formula 0 ~ 1
#f_saccade = @formula 0 ~ 1 + spl(sacc_visual_angle, 4) + spl(peak_velocity, 4) + spl(duration, 4)
f_saccade = @formula 0 ~ 1 + spl(sacc_visual_angle, 4)
# f_saccade_outside = @formula 0 ~ 1

# TODO: Test model fitting with three events
# bfDict = [
#     "02 Stimulus image shown" => (f_stimulus, bf_stimulus),
#     "saccade" => (f_saccade, bf_saccade),
#     "saccade_outside_screen" => (f_saccade_outside, bf_saccade_outside)
# ]

bfDict = [
    "02 Stimulus image shown" => (f_stimulus, bf_stimulus),
    "saccade" => (f_saccade, bf_saccade)
]

custom_solver = (X, y) -> Unfold.solver_predefined(X, y; solver=:qr) #multithreading=false)
results_all = run_unfold(
    data_df,
    bfDict;
    eventcolumn="trial_type",
    channels=["eeg"],
    solver=custom_solver,
    extract_data=prepare_eeg_data,
    windows_to_remove_all=windows_to_remove
)

save_results(results_all, data_root_path, derivatives_subfolder="Unfold/test", overwrite=false)