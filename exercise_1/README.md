# Advanced Robot Learning and Decision Making

> 🚀 Welcome! To start the exercises succesfully, read *all of the following* carefully.

## Preliminaries

- You require at least 20GB of free permament storage and 8GB (preferably 16GB) of RAM on your device.
- We strongly recommend to use Linux. It is the most widely used OS in robotics.
- We support running the exercises in our  VS Code Dev Container on Linux (recommended) and Windows 11 (WSL2). You can also setup the environment with manual installation, but you may encounter installation issues. For MacOS, most students use a VM (recommended) or manual installation.
- If you have troubles setting up the environment, ask more experienced students for help or reach out to us.
- For any issues **check the `Common Issues` section below before reaching out for help.
- :warning: **We might need to provide you with hotfixes for our code during the semester. We will communicate this as required**.

## Getting started
We will use:
- <kbd>Git</kbd> as version control system: Find a [Git introduction here](https://docs.duckietown.com/ente/devmanual-software/basics/development/git.html). Specifically, you need to know how to `clone`, `add`, `commit`, and `push`.
- <kbd>Python</kbd> as programming language: Find a [Python tutorial here](https://www.tutorialspoint.com/python/index.htm), in particular make sure to go through the basic data structures (tuples, lists, dictionaries, sets,…), loops (while, for,…), conditional statements (if-else), functions, classes and objects.
- <kbd>Docker</kbd> as environment containerization (but you won’t see it much). A [container]((https://www.docker.com/resources/what-container/)) is a standard unit of software that packages up all required code and its dependencies. <kbd>[VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)</kbd> allows us to ship such a container as a full-featured development environment that you can readily use for our exercises. Containerization is ubiquitous in modern software development: find a [Docker introduction here](https://docs.duckietown.com/ente/devmanual-software/basics/development/docker.html).
- <kbd>[VS Code](https://code.visualstudio.com/)</kbd>: Visual Studio Code provides a set of tools that speed up software development (debugging, testing, ...). Moreover, we will provide environment configurations that setup the exercise' container and make your life easier.


If they all sound completely new to you do not panic. We will require a very basic use of most of them, but it is a good time to start learning these tools since they are all widely adopted in modern robotics.

## Setting up the exercise environment
The following are the usual steps involved in setting up VS Code Devcontainers. One special feature is that we render simulations directly on the container host's display. Such display forwarding is a common failure case, is the reason why the exercise container does not work on MacOS for the moment, and explains all of the more special instructions below.

### Linux (recommended)
1. Make sure you are using a X11 Desktop session (not wayland): https://askubuntu.com/a/1516672.
2. Install [Docker](https://docs.docker.com/engine/install/), and make sure you can run Docker hello-world without error: `docker run hello-world`.
3. Install [VS Code](https://code.visualstudio.com/), with [devcontainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and [container tools extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-containers).
4. Now your local computer is setup to run the exercise container.
5. In *this project*, rename `/.devcontainer/devcontainer.linux.json` to `/.devcontainer/devcontainer.json`.
6. Open *this project* in VS code (Select File -> Open Folder). VS code should automatically detect the devcontainer and prompt you to `Reopen in container`. If not, see [here](https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container) to open it manually. **Note**: Opening the container for the first time might take up to 15 min.
8. Now you should be ready to start the exercises.

### Windows 11 (WSL2)
For windows, you require [WSL2](https://learn.microsoft.com/de-de/windows/wsl/install) to run the devcontainer, which is actually a Linux within Windows.  Here are the important steps:

1. Follow the [official installation steps (under Getting started)](https://code.visualstudio.com/blogs/2020/07/01/containers-wsl#_getting-started) to install VS Code Devcontainers in WSL2 and Docker.
   - **Note 1:** install Ubuntu 22.04 or above
   - **Note 2:** if you didnt get prompted to `enable WSL integration` by Docker as written in the above installation steps, open `Docker Desktop`, and navigate to the settings. Here, manually enable WSL integration. (There are TWO setting options for this. Make sure to enable BOTH!)
   - Make sure you can run Docker hello-world without error: `docker run hello-world`
<!--3. Install [VSCode](https://code.visualstudio.com/), with the [WSL extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl), [devcontainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and [remote dev pack](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker).-->
3. Make sure you have *this exercise code* cloned in the WSL2 file system (and not in the windows file system), for instance to `/home` (`~`). (Performance when working on the WSL file system is much better compared to Windows file system). You can access the WSL filesystem by starting a WSL2 / Ubuntu terminal.
4. In *this project*, rename `/.devcontainer/devcontainer.wsl2.json` to `/.devcontainer/devcontainer.json`.
5. Follow [the next step in the official instructions (Open VS Code in WSL 2)](https://code.visualstudio.com/blogs/2020/07/01/containers-wsl#_open-vs-code-in-wsl-2) to open this project in a VS Code devcontainer under WSL2. (Make sure to install [devcontainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and [container tools extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-containers))
6. Open *this project* in VS Code (Select File -> Open Folder). VS Code should automatically detect the devcontainer and prompt you to `Reopen in container`. If not, see [here](https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container) to open it manually. **Note**: Opening the container for the first time might take up to 15 min. 
7. Now you should be ready to start the the exercises.


### Manual installation

The following are the installation instructions for manually setting up the exercise code without using a container. While you can follow these instruction, you might run into installation issues and we can not promise to help you with them. (Manual installation might be required for MacOS, due to mujoco rendering from inside the container, display forwarding, and X11. See these issues: [1](https://gist.github.com/sorny/969fe55d85c9b0035b0109a31cbcb088), [2](https://github.com/google-deepmind/mujoco/issues/1047)).

1. Install a python environment manager, preferably [mamba](https://mamba.readthedocs.io/en/latest/) or [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/macos.html), as well as [install Git](https://git-scm.com/downloads) for MacOS.
2. For conda, the following commands set up the environment 
```
conda create --name ardm -c conda-forge python=3.11
conda activate ardm
conda install pip
pip install -e .[test,cpu,pin] // for Mac: GPU is not supported
```
3. Install [acados](https://docs.acados.org/installation/).
4. Open *this project`s* code in your favorite editor. Now you should be ready to start the the exercises.

### Using CPU or GPU
By default the containers are configured to run on the CPU(, which is sufficient for the exercises). However, you can easily configure the devcontainer to run on your [cuda-enabled GPU](https://developer.nvidia.com/cuda-gpus)  for faster training of neural networks and rendering. This is especially useful for the deep reinforcement learning exercise, and is really trendy in general :computer:. You require at least CUDA `>12.6.3`.

If you want to use the GPU for the exercise, first install the necessary [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) and [NVIDIA Container runtime](https://developer.nvidia.com/container-runtime) on your computer, if it is not installed already.

Then, to run the exercise container on the GPU:
1. In your `.devcontainer/devcontainer.json` uncomment the lines
```
"--gpus=all", // use only with GPU
"--runtime=nvidia" // use only with GPU
```
2. In `.devcontainer/Dockerfile` uncomment the GPU version, and comment out the CPU version
```
# FROM olivertum/arldm_student:cpu-0X
FROM olivertum/arldm_student:gpu-0X
```

3. Rebuild the VS Code Devcontainer (<kbd>ctrl+shift+p</kbd> > <kbd>Dev Containers: Rebuild and Reopen in Container</kbd>).

Executing the following in a terminal inside the container should now output `True` (if not, your GPU is not detected properly by pytorch):
```
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```


### Common Issues
- Common failure codes for display forwarding  include `glfw error` and `Display not found`. For fixing make sure you followed all steps mentioned above.
- If building docker container fails at `RUN apt-get update`, make sure your host systems time is set correct: https://askubuntu.com/questions/1511514/docker-build-fails-at-run-apt-update-error-failed-to-solve-process-bin-sh
- `wsl --install` stuck at 0.0%: Try [this](https://github.com/microsoft/WSL/issues/9390#issuecomment-1579398805).
- If VSCode Test Discovery is stuck during test discovery without any error: [Downgrade your python extension to 2024.20.0](https://github.com/microsoft/vscode-python/issues/24656#issue-2757930549). You can also still run the test cases from a terminal in the Docker container by running the command `pytest`.

## Solving the exercises

- Each exercise contains exactly one `exercise_0X.iypnb` file. This file will guide you through the exercise and explains everything you need to do.
- Use **Public tests** (see below) to evaluate your implementations locally on your computer, and only push to ARTEMIS once all public tests pass.
- While you can help each other setting up the environment, the programming exercise are an individual work.
- The exercises contain **programming tasks** and **exam preperation tasks**. It is crucial you solve both to be successful in the exam.

- :warning:  **Adhere to the following rules of the game, otherwise the submission system might break leading to a failed exercise:**
   - **Do not modify any code that you are not explicitely instructed to modify.**
   - **Do not rename files or functions.**
   - **Do not change function's arguments or return values.**
   - **Do not install any additional dependencies.**
   - **We check for plagiarism with automated software and human help.** Any copied code will cancel all bonus points from programming exercises for both the copier and the copied person.
- :warning: **Do not post your solutions or part of the solutions publicly available during or after the course.** This also includes course-internal communication. Please strictly adhere to this rule. Not adhering to this rule will result in exemption from the course.
- :warning: We use the equations in the **lecture notes (script)** as reference implementation for our solutions. The formulations on the slides might occasionally differ and lead to different results. Please use the lecture notes as a reference for your implementation.

## Grading and Evaluation
- The exercises are evaluated using automated code tests.
   1. **Public tests (for you)**: These tests are available to you and allow you to check your implementations.
      - These tests are included in this repository under `/test/behavior`. You can execute those tests locally as often as you want to get feedback for your implementations. Execute those tests by using the [testing feature](https://code.visualstudio.com/docs/editor/testing) in VS Code available via this icon: <img src="resources/test_icon.png" alt="test_icon" width="20"/>.
      - When you pass all tests locally, submit your code to ARTEMIS to see if all tests pass there.
      - The tests are triggered on ARTEMIS if you submit *before* the submission deadline (you can submit as often as you want before the deadline). However, during your development, we recommend running the tests locally as it is much quicker.
   2. **Hidden tests (for bonus grades)**: These tests are not available to you, and are executed only on [ARTEMIS](https://artemis.ase.in.tum.de/) when you submit your code.
      - If you fail public test cases, you will fail hidden test cases.
      - :warning: **The code that has last been submitted to ARTEMIS is used to grade your exercises.**


- :warning: **You need to solve all test cases for an exercise (or achieve 100% in ARTEMIS to pass the exercise!**
- **:warning: The relevant test cases for each exercise are also specified in ARTEMIS.**


## Submitting
Read the instructions above carefuly before submitting to ARTEMIS. Specifically, make sure that you pass the public test cases locally on your computer before submitting.

You submit to your ARTEMIS repository using standard git workflow:

1. In your repository, add the files you wish to submit (in general you need to decide, what parts of the code needs to be submitted to solve the exercises. However, as a rule of thumb, it is a good idea to submit everything that you would submit when using git for usual version control.)
```
git add ./file1
```
2. Commit 
```
git commit -m "some notes useful to you"
```

3. Pushing will commit the code to ARTEMIS, and trigger the execution of the **public** and **hidden test** cases. **Note**: perform this action *outside* of the docker container to assure you are authenticated with git.
```
git push
```


Good luck with the exercises :partying_face:!
_____
End of document.



