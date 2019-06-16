#!/bin/bash
set -uo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
CLUSTER="euler"
USER="aledan"
CORES=10
HOURS=5

mkdir -p remote

function usage() {
	echo "Usage: $0 [OPTION]..."
	echo "Communicating with clusters."

	echo -e "\nOptions: "
	printf "\t %- 30s %s\n" "-u | --user" "Specify user name(e.g. aledan)"
	printf "\t %- 30s %s\n" "-c | --cluster" "Specify cluster name(e.g. euler)"
	printf "\t %- 30s %s\n" "-p | --push" "Pushes code to cluster."
	printf "\t %- 30s %s\n" "-d | --pull" "Pull data from cluster."
	printf "\t %- 30s %s\n" "-bsub"
	printf "\t %- 30s %s\n" "-bbjobs"
	printf "\t %- 30s %s\n" "-bkill"
	printf "\t %- 30s %s\n" "-cores"
	printf "\t %- 30s %s\n" "-hours"
}

function get_id() {
	case $CLUSTER in
		euler)
			ID="$USER@euler.ethz.ch"
			MEM="rusage[mem=1024]"
			;;
		leon)
			ID="$USER@login.leonhard.ethz.ch"
			MEM="rusage[mem=1024, ngpus_excl_p=1]"
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
		rm datasets/*.npz
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

function bsub() {
	get_id
	ssh -tt $ID <<- ENDSSH
		cd NLP
		module load hdf5
		module load gcc/4.8.2 python/3.6.0
		pip3 install --user -r requirements.txt

		bsub -n $CORES -W $HOURS:00 -R "$MEM" $@

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
			-bsub)
				shift
				bsub "$@"
				exit 0
				;;
			-bbjobs)
				shift
				bbjobs "$@"
				exit 0
				;;
			-bkill)
				shift
				bkill $1
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
