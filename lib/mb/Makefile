
LIBS := -Wl,--start-group,-lxil,-lgcc,-lc,--end-group


# Add inputs and outputs from these tool invocations to the build variables 
LD_SRCS += \
	lscript.ld 

C_SRCS += \
	main.c \
	platform.c 

OBJS += \
	main.o \
	platform.o 

C_DEPS += \
	main.d \
	platform.d 
# Add inputs and outputs from these tool invocations to the build variables 
ELFSIZE += \
main.elf.size \

# All Target
all: main.elf secondary-outputs

# Tool invocations
main.elf: $(OBJS) lscript.ld $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: MicroBlaze gcc linker'
	mb-gcc -m64 -Wl,-T -Wl,lscript.ld -L/scratch/jefff/test/training-architectures/src/acdc/air/platforms/xilinx_vck190_air/workspace/project_1_wrapper/export/project_1_wrapper/sw/project_1_wrapper/domain_microblaze_0/bsplib/lib -mlittle-endian -mxl-barrel-shift -mcpu=v11.0 -mxl-soft-mul -Wl,--no-relax -Wl,--gc-sections -o "main.elf" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

main.elf.size: main.elf
	@echo 'Invoking: MicroBlaze Print Size'
	mb-size main.elf  |tee "main.elf.size"
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(EXECUTABLES)$(OBJS)$(S_UPPER_DEPS)$(C_DEPS)$(ELFSIZE) main.elf
	-@echo ' '

secondary-outputs: $(ELFSIZE)

.PHONY: all clean dependents

%.o: %.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: MicroBlaze g++ compiler'
	mb-g++ -Wall -O3 -g3 -c -fmessage-length=0 -MT"$@" -m64 -I../include -I/scratch/jefff/test/training-architectures/src/acdc/air/platforms/xilinx_vck190_air/workspace/project_1_wrapper/export/project_1_wrapper/sw/project_1_wrapper/domain_microblaze_0/bspinclude/include -mlittle-endian -mxl-barrel-shift -mcpu=v11.0 -mxl-soft-mul -Wl,--no-relax -ffunction-sections -fdata-sections -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


%.o: %.c
	@echo 'Building file: $<'
	@echo 'Invoking: MicroBlaze gcc compiler'
	mb-gcc -Wall -O3 -g3 -c -fmessage-length=0 -MT"$@" -m64 -I../include -I/scratch/jefff/test/training-architectures/src/acdc/air/platforms/xilinx_vck190_air/workspace/project_1_wrapper/export/project_1_wrapper/sw/project_1_wrapper/domain_microblaze_0/bspinclude/include -mlittle-endian -mxl-barrel-shift -mcpu=v11.0 -mxl-soft-mul -Wl,--no-relax -ffunction-sections -fdata-sections -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '
