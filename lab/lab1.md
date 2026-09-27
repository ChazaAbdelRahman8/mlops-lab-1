# MLOps Lab 1

## Question 1

After running:

```bash
uv init

the following project structure was created:

mlops-lab-1/
├── src/
│   └── mlops_lab_1/
├── .python-version
├── pyproject.toml
└── README.md
Files Created
src/mlops_lab_1/
Contains the Python source code of the project.
.python-version
Specifies the Python version used by the project.
pyproject.toml
Contains the project configuration, including the project name, version, required Python version, and dependencies.
README.md
Contains the project documentation.

uv init creates the basic structure needed to start and manage a Python project.


## Question 2

Running:

```bash
dvc init

creates the DVC configuration files and folders.

Created Files
.dvc/config
Stores the DVC project configuration. It can later contain the remote storage configuration.
.dvc/.gitignore
Tells Git to ignore DVC internal files such as temporary files and cache-related files.
.dvcignore
Tells DVC which files and folders should be ignored.
.dvc/tmp/
Contains temporary files used internally by DVC.
Which files should be pushed to Git?

The files that should be pushed to Git are:

.dvc/config
.dvc/.gitignore
.dvcignore

Temporary DVC files, cache files, and the actual large dataset should not be pushed directly to Git.

## Question 3

When using the `--global` option, the DVC credentials are stored in the user's global DVC configuration, outside the project repository.

Other configuration options are:

- `--local`: stores the configuration in `.dvc/config.local`, which is specific to the current project and should not be committed to Git.
- `--system`: stores the configuration at the system level and can be shared by users on the same computer.
- No option: stores the configuration in `.dvc/config`, which is part of the repository configuration.

Credentials such as usernames, passwords, and access tokens should never be pushed to GitHub because they are sensitive information. Only non-secret DVC configuration should be committed.

## Question 4

After running:

```bash
dvc add data
DVC created or updated the .gitignore file and added the data folder to it.

This means Git will ignore the actual dataset and will not upload the large image files directly to GitHub. Instead, DVC tracks the dataset and Git only tracks the small data.dvc pointer file.



For **Question 5**, paste this:

```markdown
## Question 5

Yes, DVC creates a `data.dvc` file after running:

```bash
dvc add data

## Question 6

On the GitHub `main` branch, the project files and DVC configuration files are present.

The actual dataset is not stored on GitHub. Instead, GitHub contains the `data.dvc` file, which acts as a pointer to the version of the dataset tracked by DVC.

The `.gitignore` file prevents the actual `data/` folder from being tracked by Git.

Therefore:

- GitHub contains the code and DVC pointer/configuration files.
- The actual dataset is not stored on GitHub.
- `data.dvc` points to the tracked version of the data.
- After running `dvc push`, the actual dataset should be available in the DagsHub DVC storage.
