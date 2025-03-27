
// Get Input Args
args = getArgument();
args = split(args,'--');

// Input and Output Paths
in_path=args[0]+"/";
out_path=args[1]+"/";

// Taken care in the python script
// File.makeDirectory(out_path);

// Get all Files in the Input Dir
filelist = getFileList(in_path);
filelist = Array.sort(filelist);

for (g = 0; g < lengthOf(filelist); g++) {
	// only use .mha files and first open them than convert them to .img files
	if(lengthOf(filelist[g])>=7)
	{
        extension= substring(filelist[g], lengthOf(filelist[g])-7,lengthOf(filelist[g]));
        basename= substring(filelist[g], 0,lengthOf(filelist[g])-7);
        name=in_path+basename;
        print("Processing "+ g+1 + "/" + lengthOf(filelist) + " : " + basename);

        if(extension==".nii.gz")
        {
            run("NIfTI-Analyze", "open="+name+".nii.gz");
            run("MHD/MHA compressed ...", "save="+out_path+basename+".mha");
        }
        run("Close All");
    }

}
run("Quit");
