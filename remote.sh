#!/bin/bash
set -uo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
CLUSTER="euler"
USER="aledan"
CORES=10
HOURS=5
CACHE="no"

mkdir -p remote

function usage() {
	echo "Usage: $0 [OPTION]..."
	echo "Communicating with clusters."

	echo -e "\nOptions: "
	printf "\t %- 30s %s\n" "-u | --user" "Specify user name(e.g. aledan)"
	printf "\t %- 30s %s\n" "-c | --cluster" "Specify cluster name(e.g. euler)"
	printf "\t %- 30s %s\n" "-p | --push" "Pushes code to cluster."
	printf "\t %- 30s %s\n" "-d | --pull" "Pull data from cluster."
	printf "\t %- 30s %s\n" "-bbjobs"
	printf "\t %- 30s %s\n" "-cpu [command]" "Send cpu jobs."
	printf "\t %- 30s %s\n" "-gpu [command]" "Send gpu jobs."
	printf "\t %- 30s %s\n" "-bkill [job_id]"
	printf "\t %- 30s %s\n" "-bpeek [job_id]"
	printf "\t %- 30s %s\n" "-cores [num]"
	printf "\t %- 30s %s\n" "-hours [num]"
	printf "\t %- 30s %s\n" "-cache" "Use cached training and validation datasets."
}

function get_id() {
	case $CLUSTER in
		euler)
			ID="$USER@euler.ethz.ch"
			MEM="rusage[mem=1024]"
			PYTHON_MODULE=python/3.6.0
			;;
		leon)
			ID="$USER@login.leonhard.ethz.ch"
			MEM="rusage[mem=1024]"
			PYTHON_MODULE=python/3.6.1
			;;
		* )
			echo "Wrong cluster name. Use either 'euler' or 'leon'."
			exit 1
	esac
}

function push_source() {
	get_id
	BRANCH=`git branch | grep \* | cut -d ' ' -f2`
	# sync current branch
	ssh -tt $ID <<- ENDSSH
		cd NLP
		git stash
		git checkout $BRANCH
		git pull
		if [ "$CACHE" == "no" ]; then
			rm datasets/*.npz
		fi
		exit
	ENDSSH
	# copy local modifications
	FILES=`git ls-files`
	scp $FILES $ID:NLP
}

function pull_data() {
	get_id

	mkdir -p remote/$CLUSTER

	# sync data files
	rsync -h -v -r -P -t $ID:NLP/logs remote/$CLUSTER
	rsync -h -v -r -P -t $ID:NLP/models remote/$CLUSTER
	rsync -h -v -r -P -t $ID:NLP/checkpoints remote/$CLUSTER
}

function cpu_send() {
	get_id
	ssh -tt $ID <<- ENDSSH
		cd NLP
		module load hdf5
		module load gcc/4.8.2 $PYTHON_MODULE
		pip3 install --user -r requirements.txt
        python3 -c "import nltk; nltk.download('wordnet')"

		bsub -n $CORES -W $HOURS:00 -R "$MEM" $@

		exit
	ENDSSH
}

function gpu_send() {
	ssh -tt $USER@login.leonhard.ethz.ch <<- ENDSSH
		cd NLP
		module load python_gpu/3.7.1 cuda/10.0.130
		module load gcc/6.3.0
		module load hdf5/1.10.1

		pip3 install --user -r requirements.txt
        python3 -c "import nltk; nltk.download('wordnet')"

		bsub -W $HOURS:00 -n $CORES -R "rusage[mem=4000,scratch=5000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" "$@"

		exit
	ENDSSH
}

function bbjobs() {
	get_id
	ssh -tt $ID <<- ENDSSH
		bbjobs
		exit
	ENDSSH
}

function bpeek() {
	get_id
	ssh -tt $ID <<- ENDSSH
		bpeek $1
		exit
	ENDSSH
}

function bkill() {
	get_id
	ssh -tt $ID <<- ENDSSH
		echo "Killing $1"
		bkill $1
		exit
	ENDSSH
}

function parse_command_line_options() {
	while [ "${1:-}" != "" ]; do
		case $1 in
			-u | --user)
				shift
				USER=$1
				;;
			-c | --cluster)
				shift
				CLUSTER=$1
				;;
			-cores)
				shift
				CORES=$1
				;;
			-hours)
				shift
				HOURS=$1
				;;
			-cache)
				CACHE="yes"
				;;
			-cpu)
				shift
				cpu_send "$@"
				exit 0
				;;
			-gpu)
				shift
				gpu_send "$@"
				exit 0
				;;
			-bbjobs)
				shift
				bbjobs
				exit 0
				;;
			-bkill)
				shift
				bkill $1
				exit 0
				;;
			-bpeek)
				shift
				bpeek $1
				exit 0
				;;
			-p | --push)
				shift
				push_source $CLUSTER
				exit 0
				;;
			-d | --pull)
				shift
				pull_data $CLUSTER
				exit 0
				;;
			* )
				usage
				exit 1
		esac

		shift
	done
	usage
}

pushd $DIR > /dev/null
parse_command_line_options "$@"
popd > /dev/null
