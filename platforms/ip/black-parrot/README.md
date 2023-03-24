# Overview

This folder enables the user to package a BlackParrot RISC-V core as a Vivado IP block for
use in custom block designs.

# Setup Instructions

Before creating and packagin the BlackParrot IP block, you must fetch the BlackParrot RTL.
The following commands will fetch and patch the BlackParrot RTL in preparation for IP packaging.

```
make checkout # fetch BlackParrot RTL as git submodules
make patch_bp # apply patches to BlackParrot RTL to create a suitable core config and top-level files
```

# Build Instructions

First, source the Vivado tools into your environment using the Vivado settings64.sh script.
The BlackParrot IP build script was designed and tested with Vivado 2021.2.

Then, run the following commands to create and package the BlackParrot IP block. These commands
will generate a `.tar.gz` compressed archive of the BlackParrot IP directory.

```
make build_ip # create the BlackParrot IP project and export the IP
make pack_ip # create the BlackParrot IP .tar.gz archive
```

# BlackParrot SDK

The BlackParrot processor uses a port of the RISC-V GNU Toolchain to compile programs.
This compiler is included as part of the BlackParrot SDK. To download and extract the BlackParrot
SDK, run the following commands.

```
make get_sdk # download and extract the BlackParrot SDK
```

# Adding BlackParrot IP to a project

Adding the BlackParrot IP to your project and block design is as simple as adding a copy of the
BlackParrot IP directory as an additional IP repository in Vivado and then creating an instance
of the IP in the project's block design.

The BlackParrot IP `.tar.gz` archive is meant to be copied to and extracted at a suitable location
for your project, before adding the extracted folder as an IP repository in the project.

-----

<p align="center">Copyright&copy; 2022 Advanced Micro Devices, Inc.</p>
