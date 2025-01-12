# Welcome to Computer Vision!

This is an **ungraded** project will help you get up and running with a working environment! For those that are familiar with this setup -- let's rehash anyways.
We'll focus on install a few crucial tools to help us maintain consistent environments between students.

- [Visual Studio Code](https://code.visualstudio.com/Download)
- [Git Bash (Only necessary for Windows)](https://git-scm.com/downloads)
- [Miniforge](https://github.com/conda-forge/miniforge)

## Visual Studio Code

[Visual Studio Code](https://code.visualstudio.com/Download) is **recommended** for this course for a few reasons.

- Consistent development environment (so the teaching staff can more easily help!)
- Ease of integration with Jupyter Notebooks and `git`.

### macOS Users (Optional)

`brew` is a very useful package management tool for `macOS`. It can make some of the above installation even simpler.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install --cask visual-studio-code
brew install gh
```

## Visual Studio Code Extensions

Download the following extensions:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Remote Development Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)

*Note: confirm that all the extensions are installed/enabled before continuing. 

## Environment Setup

We will be using Mamba/Miniforge to create a consistent Python environment for running and testing our code. For each project, we will create a new conda environment with slightly different packages (specified in `environment.yml`). You will need to install Miniforge before you can start working in these environments.

See the [Miniforge GitHub repository](https://github.com/conda-forge/miniforge) and scroll download to the downloads section to install the correct version of Miniconda for your machine. See the [Mamba website](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html#mamba) for more information on system requirements and what Mamba does.

For Windows only: This step only needs to be done once. Initialize mamba for powershell by running: `mamba init powershell` in the Miniforge Prompt.

## Setting up Project 0

In VSCode, open the unzipped project folder. In the `conda` folder, you will see an installation script, `install.sh`. Running this script will create your `conda` environment, which should then be activated. On macOS, you can run this script by navigating to the `conda` directory in a terminal and running `./install.sh.` On Windows, use the Miniforge Prompt instead of the built-in terminal and run the command `.\install.sh` instead. Doing so will require [Git Bash](https://git-scm.com/downloads).

If you do not see `cv_proj0` at the start of each line of your VSCode command prompt, you will need to manually activate your environment with the following commands from the root of your repository.

```bash
conda activate cv_proj0
pip install -e .
```

If the install script does not work for you, you may alternatively run the following set of commands from the `project-0` directory.

```
cd conda
mamba env create -f environment.yml
cd ..
conda activate cv_proj0
pip install -e .
```

**Important Note**: Once you have created the project environment, you'll need to select it as the `Default Interpreter`. Open up the Command Palette and search for `Python: Create Environment`. Then select the environment you just created from the menu.

## Unit Test Setup

We will use `pytest` in this class as a testing framework. This will run unit tests on your code to verify correctness. Some of the unit tests will given to you as a courtesy; however, others will be run at submission time.

### Run the unit tests

The following command will run all unit tests in the `tests` folder.

```bash
pytest tests
```

The unit tests will initially fail -- modify `src/vision/linalg.py` and see if you can pass the tests!

## Jupyter Notebook

Each project (except this one) will have a jupyter notebook which will allow you to run and test your code in an interactive Python coding environment. Jupyter notebook should already be installed, but if not you can visit [this page](https://jupyter.org/install) to install.

You can open these notebooks directly in VSCode or use the following command to start a jupter notebook session:
```bash
jupyter notebook ./<file_name>.ipynb
```

Once in the notebook, you can edit text cells by double-clicking on them and edit/run code cells (use the shortcut cmd+enter or shift+enter to run code cells).

## Submission
To create the zip file to upload on gradescope, run the following from within your conda environment.

```
python submission.py --gt_username your_username
```

# Conclusion

We hope this project helped you get your environment setup for this course. If you have any questions / concerns, please post on [piazza](https://piazza.com).
