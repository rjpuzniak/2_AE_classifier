script_path=$(pwd)

datasets='ABIDE Athletes CHIASM COBRE HCP Leipzig MCIC UoN'


for dataset in $datasets; do

	echo $dataset

	cd $dataset/

	for subject in */; do

		if [ ! -f $subject/chiasm.nii.gz ]; then

			echo $subject
			rm $subject/*

		fi

	done


	cd ..

done

cd $script_path











