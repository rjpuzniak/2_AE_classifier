for i in HCP/*/chiasm.nii.gz; do

	mrview ${i::-13}t1w_1mm_iso.nii.gz $i

	read -p 'Keep?' yn
	case $yn in
		[Yy] ) continue;;
		[Nn] ) rm $i; echo $i >> removed_HCP.txt; continue;;
	esac

done




