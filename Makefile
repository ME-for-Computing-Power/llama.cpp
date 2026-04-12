SHELL := /usr/bin/env bash

IMAGE ?= fpai-icraft:latest
SOURCE_DIR ?= $(CURDIR)
SOURCE_DIR_ABS := $(abspath $(SOURCE_DIR))
GENERATOR ?= Ninja
BUILD_TYPE ?= Release
JOBS ?= $(shell nproc)
DOCKER_ARGS ?=

FMSH_ZG330_TARGET ?= ggml-fmsh_zg330
FMSH_ZG330_ARCH ?= x86_64
FMSH_ZG330_ARCH_FLAVOR := $(FMSH_ZG330_ARCH)
ifneq (,$(filter x86_64 amd64 x64 x86,$(FMSH_ZG330_ARCH)))
FMSH_ZG330_ARCH_FLAVOR := x64
endif
ifneq (,$(filter aarch64 arm64 armv8% arm%,$(FMSH_ZG330_ARCH)))
FMSH_ZG330_ARCH_FLAVOR := arm64
endif

FMSH_ZG330_BUILD_X64 ?= build-fmsh-zg330-x64
FMSH_ZG330_BUILD_ARM64 ?= build-fmsh-zg330-arm64
FMSH_ZG330_BUILD_DIR ?= $(if $(filter arm64,$(FMSH_ZG330_ARCH_FLAVOR)),$(FMSH_ZG330_BUILD_ARM64),$(if $(filter x64,$(FMSH_ZG330_ARCH_FLAVOR)),$(FMSH_ZG330_BUILD_X64),build-fmsh-zg330-$(FMSH_ZG330_ARCH_FLAVOR)))
FMSH_ZG330_BUILD_DIR_ABS := $(abspath $(FMSH_ZG330_BUILD_DIR))

FMSH_ZG330_MODELZOO_ROOT ?= /ModelzooDeps
FMSH_ZG330_TOOLCHAIN_FILE ?=
FMSH_ZG330_EXTRA_CMAKE_ARGS ?=
FMSH_ZG330_USE_DOCKER ?= 1
FMSH_ZG330_DOCKER_ARGS ?= --network host
FMSH_ZG330_CONFIGURE_ONLY ?= 0
FMSH_ZG330_NO_INSTALL ?= 1
FMSH_ZG330_INSTALL_PREFIX ?= $(FMSH_ZG330_BUILD_DIR)/install
FMSH_ZG330_DEBUG_SYMBOLS ?= on
FMSH_ZG330_DEBUG_FLAGS ?= -g

FMSH_ZG330_IP ?= 192.168.110.157
FMSH_ZG330_PORT ?= 9981

SRC_MOUNT := /workspace/src
BUILD_MOUNT := /workspace/build
INNER_SCRIPT := $(SRC_MOUNT)/docker-cross-build-inner.sh

.PHONY: help dev-shell \
	fmsh-zg330-configure fmsh-zg330-build fmsh-zg330-install \
	fmsh-zg330-configure-x64 fmsh-zg330-build-x64 \
	fmsh-zg330-configure-arm64 fmsh-zg330-build-arm64 \
	fmsh-zg330-run-x64-zg330 fmsh-zg330-clean fmsh-zg330-dry-run

help:
	@echo "Targets:"
	@echo "  make fmsh-zg330-configure        Configure in Docker"
	@echo "  make fmsh-zg330-build            Configure + build in Docker"
	@echo "  make fmsh-zg330-install          Configure + build + install in Docker"
	@echo "  make fmsh-zg330-configure-x64    Configure x64 build"
	@echo "  make fmsh-zg330-build-x64        Build x64 backend"
	@echo "  make fmsh-zg330-configure-arm64  Configure arm64 cross build"
	@echo "  make fmsh-zg330-build-arm64      Build arm64 backend"
	@echo "  make fmsh-zg330-run-x64-zg330    Print x64 socket-mode runtime env"
	@echo "  make fmsh-zg330-dry-run          Preview docker command"
	@echo "  make dev-shell                   Enter dev container"
	@echo
	@echo "Common vars: IMAGE GENERATOR BUILD_TYPE JOBS DOCKER_ARGS"
	@echo "Backend vars: FMSH_ZG330_ARCH FMSH_ZG330_BUILD_DIR FMSH_ZG330_MODELZOO_ROOT"
	@echo "              FMSH_ZG330_TOOLCHAIN_FILE FMSH_ZG330_EXTRA_CMAKE_ARGS"
	@echo "Runtime vars: FMSH_ZG330_IP FMSH_ZG330_PORT"

dev-shell:
	docker run -it --rm -v "$(CURDIR)":/workspace $(IMAGE)

ifeq ($(FMSH_ZG330_USE_DOCKER),1)

fmsh-zg330-configure:
	@$(MAKE) fmsh-zg330-run FMSH_ZG330_CONFIGURE_ONLY=1 FMSH_ZG330_NO_INSTALL=1

fmsh-zg330-build:
	@$(MAKE) fmsh-zg330-run FMSH_ZG330_CONFIGURE_ONLY=0 FMSH_ZG330_NO_INSTALL=1

fmsh-zg330-install:
	@$(MAKE) fmsh-zg330-run FMSH_ZG330_CONFIGURE_ONLY=0 FMSH_ZG330_NO_INSTALL=0

else

fmsh-zg330-configure:
	@$(MAKE) _fmsh-zg330-run FMSH_ZG330_CONFIGURE_ONLY=1 FMSH_ZG330_NO_INSTALL=1

fmsh-zg330-build:
	@$(MAKE) _fmsh-zg330-run FMSH_ZG330_CONFIGURE_ONLY=0 FMSH_ZG330_NO_INSTALL=1

fmsh-zg330-install:
	@$(MAKE) _fmsh-zg330-run FMSH_ZG330_CONFIGURE_ONLY=0 FMSH_ZG330_NO_INSTALL=0

endif

fmsh-zg330-configure-x64:
	@$(MAKE) fmsh-zg330-configure FMSH_ZG330_ARCH=x86_64 FMSH_ZG330_BUILD_DIR="$(FMSH_ZG330_BUILD_X64)"

fmsh-zg330-build-x64:
	@$(MAKE) fmsh-zg330-build FMSH_ZG330_ARCH=x86_64 FMSH_ZG330_BUILD_DIR="$(FMSH_ZG330_BUILD_X64)"

fmsh-zg330-configure-arm64:
	@$(MAKE) fmsh-zg330-configure FMSH_ZG330_ARCH=aarch64 FMSH_ZG330_BUILD_DIR="$(FMSH_ZG330_BUILD_ARM64)"

fmsh-zg330-build-arm64:
	@$(MAKE) fmsh-zg330-build FMSH_ZG330_ARCH=aarch64 FMSH_ZG330_BUILD_DIR="$(FMSH_ZG330_BUILD_ARM64)"

_fmsh-zg330-run:
	@mkdir -p "$(FMSH_ZG330_BUILD_DIR_ABS)"
	FMSH_ZG330_SOURCE_DIR='$(SOURCE_DIR_ABS)' \
	FMSH_ZG330_BUILD_DIR='$(FMSH_ZG330_BUILD_DIR_ABS)' \
	FMSH_ZG330_GENERATOR='$(GENERATOR)' \
	FMSH_ZG330_BUILD_TYPE='$(BUILD_TYPE)' \
	FMSH_ZG330_ARCH='$(FMSH_ZG330_ARCH)' \
	FMSH_ZG330_BUILD_TARGETS='$(FMSH_ZG330_TARGET)' \
	FMSH_ZG330_DEBUG_SYMBOLS='$(FMSH_ZG330_DEBUG_SYMBOLS)' \
	FMSH_ZG330_INSTALL_PREFIX='$(abspath $(FMSH_ZG330_INSTALL_PREFIX))' \
	FMSH_ZG330_JOBS='$(JOBS)' \
	FMSH_ZG330_CONFIGURE_ONLY='$(FMSH_ZG330_CONFIGURE_ONLY)' \
	FMSH_ZG330_NO_INSTALL='$(FMSH_ZG330_NO_INSTALL)' \
	FMSH_ZG330_TOOLCHAIN_FILE='$(FMSH_ZG330_TOOLCHAIN_FILE)' \
	FMSH_ZG330_MODELZOO_ROOT='$(FMSH_ZG330_MODELZOO_ROOT)' \
	FMSH_ZG330_EXTRA_CMAKE_ARGS='$(FMSH_ZG330_EXTRA_CMAKE_ARGS)' \
	CMAKE_C_FLAGS='$(FMSH_ZG330_DEBUG_FLAGS)' \
	CMAKE_CXX_FLAGS='$(FMSH_ZG330_DEBUG_FLAGS)' \
	bash "$(CURDIR)/docker-cross-build-inner.sh"

fmsh-zg330-run:
	@mkdir -p "$(FMSH_ZG330_BUILD_DIR_ABS)"
	@install_prefix_container='$(FMSH_ZG330_INSTALL_PREFIX)'; \
	case "$$install_prefix_container" in \
	  '$(FMSH_ZG330_BUILD_DIR_ABS)'/*) install_prefix_container='$(BUILD_MOUNT)'/$${install_prefix_container#'$(FMSH_ZG330_BUILD_DIR_ABS)'/} ;; \
	  '$(SOURCE_DIR_ABS)'/*) install_prefix_container='$(SRC_MOUNT)'/$${install_prefix_container#'$(SOURCE_DIR_ABS)'/} ;; \
	  '$(FMSH_ZG330_BUILD_DIR)'/*) install_prefix_container='$(BUILD_MOUNT)'/$${install_prefix_container#'$(FMSH_ZG330_BUILD_DIR)'/} ;; \
	  '$(SOURCE_DIR)'/*) install_prefix_container='$(SRC_MOUNT)'/$${install_prefix_container#'$(SOURCE_DIR)'/} ;; \
	esac; \
	toolchain_file_container='$(FMSH_ZG330_TOOLCHAIN_FILE)'; \
	if [[ -n "$$toolchain_file_container" ]]; then \
	  if [[ "$$toolchain_file_container" = /* ]]; then \
	    case "$$toolchain_file_container" in \
	      '$(SOURCE_DIR_ABS)'/*) toolchain_file_container='$(SRC_MOUNT)'/$${toolchain_file_container#'$(SOURCE_DIR_ABS)'/} ;; \
	      '$(FMSH_ZG330_BUILD_DIR_ABS)'/*) toolchain_file_container='$(BUILD_MOUNT)'/$${toolchain_file_container#'$(FMSH_ZG330_BUILD_DIR_ABS)'/} ;; \
	    esac; \
	  else \
	    toolchain_file_container='$(SRC_MOUNT)'/$$toolchain_file_container; \
	  fi; \
	fi; \
	docker run --rm \
	  $(FMSH_ZG330_DOCKER_ARGS) \
	  $(DOCKER_ARGS) \
	  --user "$$(id -u):$$(id -g)" \
	  --mount "type=bind,src=$(SOURCE_DIR_ABS),dst=$(SRC_MOUNT),ro" \
	  --mount "type=bind,src=$(FMSH_ZG330_BUILD_DIR_ABS),dst=$(BUILD_MOUNT),readonly=false" \
	  -w "$(BUILD_MOUNT)" \
	  --entrypoint bash \
	  -e FMSH_ZG330_SOURCE_DIR="$(SRC_MOUNT)" \
	  -e FMSH_ZG330_BUILD_DIR="$(BUILD_MOUNT)" \
	  -e FMSH_ZG330_GENERATOR="$(GENERATOR)" \
	  -e FMSH_ZG330_BUILD_TYPE="$(BUILD_TYPE)" \
	  -e FMSH_ZG330_ARCH="$(FMSH_ZG330_ARCH)" \
	  -e FMSH_ZG330_BUILD_TARGETS="$(FMSH_ZG330_TARGET)" \
	  -e FMSH_ZG330_DEBUG_SYMBOLS="$(FMSH_ZG330_DEBUG_SYMBOLS)" \
	  -e CMAKE_C_FLAGS="$(FMSH_ZG330_DEBUG_FLAGS)" \
	  -e CMAKE_CXX_FLAGS="$(FMSH_ZG330_DEBUG_FLAGS)" \
	  -e FMSH_ZG330_INSTALL_PREFIX="$$install_prefix_container" \
	  -e FMSH_ZG330_JOBS="$(JOBS)" \
	  -e FMSH_ZG330_CONFIGURE_ONLY="$(FMSH_ZG330_CONFIGURE_ONLY)" \
	  -e FMSH_ZG330_NO_INSTALL="$(FMSH_ZG330_NO_INSTALL)" \
	  -e FMSH_ZG330_TOOLCHAIN_FILE="$$toolchain_file_container" \
	  -e FMSH_ZG330_MODELZOO_ROOT="$(FMSH_ZG330_MODELZOO_ROOT)" \
	  -e FMSH_ZG330_EXTRA_CMAKE_ARGS="$(FMSH_ZG330_EXTRA_CMAKE_ARGS)" \
	  "$(IMAGE)" -lc "$(INNER_SCRIPT)"

fmsh-zg330-run-x64-zg330:
	@echo "Built x64 backend should use socket mode."
	@echo "export GGML_FMSH_ZG330_IP=$(FMSH_ZG330_IP)"
	@echo "export GGML_FMSH_ZG330_PORT=$(FMSH_ZG330_PORT)"
	@echo "export GGML_FMSH_ZG330_CACHE_DIR=$(CURDIR)/.cache/ggml-fmsh_zg330"

fmsh-zg330-dry-run:
	@echo docker run --rm $(FMSH_ZG330_DOCKER_ARGS) $(DOCKER_ARGS) -v "$(CURDIR)":/workspace $(IMAGE) ...

fmsh-zg330-clean:
	rm -rf "$(FMSH_ZG330_BUILD_X64)" "$(FMSH_ZG330_BUILD_ARM64)"
