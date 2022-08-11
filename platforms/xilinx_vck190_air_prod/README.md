# vck190 AIR platform (Vitis 2021.2, Ubuntu 18.04 filesystem) 

At the moment, the build flow for building the AIR platform targeting the vck190 production board involves a few manual steps to generate the final sd image. This process will be streamlined in the near future but for now, it involves the following.

1. Build petalinux-based platform with custom Vivado design (AIE NoC connections, BRAM off NoC, microblaze and AIR data mover). A generated .xsa and .bsp file is copied into a PYNQ compliant platform folder (`pynq/vck190_air`). Be sure the machine used meets software requirements to run vivado/ vitis/ petalinux.

2. Build custom PYNQ/ Ubuntu filesystem which takes in the previously generated platform folder. Build step requires a machine with root/sudo access in order for some mount/fs commands to run. This generates an sd image file (sd_card.img) with two partitions, a primary fat32 boot partition and an ext4 filesystem partition.

3. **(Optional)** Copy BOOT.BIN on the primary partition in order to pick up the microblaze reset boot sequence.

4. **(Optional)** Copy libxaiengine.so.* files to /usr/lib in the event PYNQ built libxaiengine libraries do not work properly.

## Step 1 - Build petalinux-base platform
This step should be relatively straighforward. Navigate to this current folder (`platforms/xilinx_vck19_air_prod`) and then call make.
```
cd mlir-air/platforms/xilinx_vck190_air_prod
make pynq
```
This should generate the .xsa and .bsp files and copy them to `pynq/vck190_air`. Note that this flow also generates the full boot files (BOOT.BIN) which will be found under `platforms/xilinx_vck190_air_prod/bootgen/BOOT.BIN`. This can be used to replace the final BOOT.BIN on the boot partition of the sd_card.img file.

## Step 2 - Build custom PYNQ/ Ubuntu filesystem 
This step involves cloning another gitenterprise repo but since it's connected to this build flow, we will describe the build step here.

We will clone the custom PYNQ fork, download and copy the base bionic Ubuntu image and then run make. These build instructions are based off the ones described in Jeff's PYNQ fork [here](https://gitenterprise.xilinx.com/XRLabs/mlir-air/blob/main/docs/vck190_building_pynq.md). 

Copy the refernce bionic Ubuntu base image from `/group/xrlabs2/pynq/public/v2.6.0_images/bionic.aarch64.2.6.0_2020_09_21.zip` and unzip it to `sdbuild/output` which you create.
```
git clone https://gitenterprise.xilinx.com/jackl/PYNQ.git
cd PYNQ/sdbuild
mkdir output
cp <unzipped bionic img file> ./output/.
```
Now you can run the make command with references to the PYNQ platform folder generated in Step 1.
```
make PREBUILT=output/bionic.aarch64.*.img BOARDDIR=<path to mlir-air/pynq folder> nocheck_images 
```
You may need to intervene in the beginning and in the middle to type in your password in order to run sudo fs commands.

Now the final sd image file wil be saved to `mlir-air/sdbuild/output/bionic.img` which you can image onto an sd card and boot your vck190 production board with.

---

## Step 3 - **(Optional)** Copy BOOT.BIN
As mentioned in Step 1, you can replace the boot file in your sd card's primary parition with the one from  `platforms/xilinx_vck190_air_prod/bootgen/BOOT.BIN`. You can either mount the image and overwrite it or plug the sd card into your host machine and copy the new file in. 

## Step 4 - **(Optional)** Copy libxaiengine.so.* libs
It may be the case that the PYNQ compiled libxaiengine libs (which is compiled from source) doesn't work on the board, in which case, you can copy over the ones created by the mlir-aie bare production board. Copy the libxaiengine.so.* to the secondary parition (ubuntu fs) under /opt/xaiengine/lib.x
