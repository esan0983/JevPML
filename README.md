# JevPML
A Jev-based project that automates a model selection + basic preprocessing + ML pipeline. Currently working on Logistic Regression.
## Commit Notes (11/5)
* Smooth pipeline from state creation to logistic regression succeeded, barring two assumptions checks
## To-Do
* VIF can't work due to the categorical encoding resulting into a divide by zero error
* Find ways to automatically manipulate datasets if assumptions fail
* Be more transparent to the user by telling them what was removed, what passes, what fails, etc.
* Document what happens behind the scenes in the README