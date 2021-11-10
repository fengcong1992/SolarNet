#!/bin/bash
#SBATCH --account=solfor
#SBATCH --time=2-00:00:00
#SBATCH --job-name=WT17I14
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=cong.feng@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=RJob.%j.out  # %j will be replaced with the job ID
#SBATCH --partition=standard

Rscript /lustre/eaglefs/projects/solfor/perform/ERCOT_data/WeatherForecast/2017/IntraDayWeatherForecasting17_Eagle_byStation.R 14