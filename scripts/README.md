Scripts etc. - everything that runs something on it's own and is not a function, goes here.

- `exp-startup` - bash script to create the BIDS-style `data/sub-xxx/ses-yyy/beh` folder for a participant, copy the behavioural file templates from the `assets` folder and rename them according to the subject.



Please use easily understandable filenames

E.g.
```
Fig1_simulateSimulation.jl
Results_analysisOfHairColor.jl
Results_tableResultSizeCalculation.jl
```

or what could also work
```
01_loadAndPreprocessData.jl
02_generateFilteredVersion.jl
03_analyseERPsForSaccadeAmplitude.jl
```

But please choose one way to present your scripts and keep with it. If you wonder if it is a good idea to edit this Readme.md to decsribe what each script does. **Yes it is!**
