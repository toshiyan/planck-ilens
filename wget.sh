#!/bin/bash

dir="data/"

check_file(){
	FILE=${dir}${1}
	if [ ! -f "${FILE}" ]; then
		wget -O ${FILE} "${2}"
	else
		echo "file exist: ${FILE}"
	fi
}

# //// DR2 //// #
# CO bias map 
#check_file HFI_BiasMap_100-CO-nominal_2048_R3.00_full.fits http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_BiasMap_100-CO-nominal_2048_R3.00_full.fits
#check_file HFI_BiasMap_217-CO-nominal_2048_R3.00_full.fits http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_BiasMap_217-CO-nominal_2048_R3.00_full.fits
#check_file HFI_BiasMap_353-CO-nominal_2048_R3.00_full.fits http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_BiasMap_353-CO-nominal_2048_R3.00_full.fits

#check_file HFI_BiasMap_100-CO-noiseRatio_2048_R3.00_full.fits http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_BiasMap_100-CO-noiseRatio_2048_R3.00_full.fits
#check_file HFI_BiasMap_217-CO-noiseRatio_2048_R3.00_full.fits http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_BiasMap_217-CO-noiseRatio_2048_R3.00_full.fits
#check_file HFI_BiasMap_353-CO-noiseRatio_2048_R3.00_full.fits http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_BiasMap_353-CO-noiseRatio_2048_R3.00_full.fits

check_file COM_CMB_IQU-smica_2048_R3.00_full.fits "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CMB_IQU-smica_2048_R3.00_full.fits"
check_file COM_CMB_IQU-nilc_2048_R3.00_full.fits "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CMB_IQU-nilc_2048_R3.00_full.fits"

