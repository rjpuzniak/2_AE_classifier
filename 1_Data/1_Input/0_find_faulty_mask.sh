for i in */*/chiasm.nii.gz; do
	
	max=$(mrstats -output max $i -quiet)

	if [ $max == 0 ]; then

		echo $i

	fi

done
