from pip._internal.cli.main import main
from os import listdir
from os.path import isfile, join
import os
import sys
from sys import platform

if sys.version_info[0] < 3:
    print("This is not Python3 - Not support!!!")
    exit(1)
else:
    check_version = False
    check_os = False

    # Get configuration of library from file
    file_config = open(".configuration", "r")
    data = file_config.read()
    file_config.close()
    configuration = data.split("\n")

    PYTHON_LIBRARY_VERSION = configuration[0]
    PYTHON_LIBRARY_OS = configuration[1]

    # Detect system
    detect_os = ""
    if platform == "linux" or platform == "linux2":
        detect_os = "Linux"
    elif platform == "darwin":
        detect_os = "MacOS"
    elif platform == "win32":
        detect_os = "Windows"

    detect_python = str(sys.version_info.major) + "." + str(sys.version_info.minor)

    if platform == PYTHON_LIBRARY_OS:
        check_os = True

    if detect_python == PYTHON_LIBRARY_VERSION:
        check_version = True

    # Start install
    if check_version is True and check_os is True:
        if platform == "win32":
            # Get directory of library
            dir_path = os.path.dirname(os.path.realpath(__file__))
            current_dir = os.path.basename(dir_path)

            # Get library list name
            packpath = os.path.basename(str(current_dir) + "\libs")
            packages = [(packpath + '/' + f) for f in listdir(packpath) if isfile(join(packpath, f))]

            libpath = os.path.basename(str(current_dir) + "\envLibs")
            libs = [(libpath + '/' + f) for f in listdir(libpath) if isfile(join(libpath, f))]

            try:
                print("Setting environment...")
                print("==================================")
                main(['install', '--no-index', '--find-links="' + current_dir + '\envLibs"'] + libs)

                print("Installing packages...")
                print("==================================")
                main(['install', '--no-index', '--find-links="' + current_dir + '\libs"'] + packages)

            except Exception as e:
                print("==================================")
                print("Failed to install packages.")
                exit(1)

            exit(0)

        elif platform == "darwin":
            print("This is not supporting on MacOS")
            exit(1)
        else:
            print("This is not supporting on Linux")
            exit(1)

    else:
        print("==================================")
        if check_os is False:
            if detect_os == "":
                print("SYSTEM - OS_PLATFORM: cannot detect")
            else:
                print("SYSTEM - OS_PLATFORM: " + str(detect_os))
        if check_version is False:
            if len(detect_python) < 3:
                print("SYSTEM - PYTHON_VERSION: cannot detect")
            else:
                print("SYSTEM - PYTHON_VERSION: " + str(detect_python))
        exit(1)