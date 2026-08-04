function construct_subject_bids_path(
    bids_root::AbstractString,
    subject_id::Union{Int,AbstractString},
    derivative_dir::Union{AbstractString,Nothing}=nothing;
    session::Union{AbstractString,Nothing}=nothing,
    datatype::AbstractString="eeg",
)

    subject_string = isa(subject_id, String) ? "sub-"*subject_id : "sub-"*lpad(subject_id, 3, "0")
    session_string = !isnothing(session) ? "ses-"*session : ""

    if !isnothing(derivative_dir)
        path = joinpath(bids_root, "derivatives", derivative_dir, subject_string, session_string, datatype)
    else
        path = joinpath(bids_root, subject_string, session_string, datatype)
    end

    return path

end

function construct_subject_filename(
    subject_id::Union{Int,AbstractString},
    name::AbstractString,
    file_extension::AbstractString;
    session::Union{AbstractString,Nothing}=nothing,
    run::Union{Int,AbstractString,Nothing}=nothing,
    task::Union{AbstractString,Nothing}=nothing,
)

    subject_string = isa(subject_id, String) ? "sub-"*subject_id*"_" : "sub-"*lpad(subject_id, 3, "0")*"_"
    session_string = !isnothing(session) ? "ses-"*session*"_" : ""
    run_string = !isnothing(run) ? "run-"*string(run)*"_" : ""
    task_string = !isnothing(task) ? "task-"*task*"_" : ""

    return join([subject_string, session_string, task_string, run_string, name, ".", file_extension], "")

end

# Function to determine whether a gaze point (x,y) is outside of the screen
function outside_screen(x, y, screen_resolution=(1920, 1080))
    return (x < 0) || (x > screen_resolution[1]) || (y < 0) || (y > screen_resolution[2])
end