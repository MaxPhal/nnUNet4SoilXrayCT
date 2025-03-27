
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
	extension= substring(filelist[g], lengthOf(filelist[g])-3,lengthOf(filelist[g]));
	basename= substring(filelist[g], 0,lengthOf(filelist[g])-4);
	name=in_path+basename;

		if(extension=="tif" &&
		!File.exists(out_path+basename+".img") &&
		!File.exists(out_path+basename+".nii.gz"))
		{
			print("Processing "+ g+1 + "/" + lengthOf(filelist) + " : " + basename);
			open(name+".tif");
			run("Analyze... ", "save="+out_path+basename+".img");
		}
		else
		{
		    print("Processing "+ g+1 + "/" + lengthOf(filelist) + ": " + basename +" - File already exists");
		}
	run("Close All");

}
run("Quit");
