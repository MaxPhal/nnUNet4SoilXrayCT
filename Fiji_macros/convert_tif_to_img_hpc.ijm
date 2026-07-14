/* Notes
 * For cluster computation, the name of each file is given via SLURM_array_TASK_ID
*/

// Get Input Args
args = getArgument();
args = split(args,'--');

// Input and Output Paths 
in_path=args[0];
out_path=args[1]+"/";

// print("Arg 1:" + in_path);
// print("Arg 2:" + out_path);

// Retrieve sample name
dotPos= lastIndexOf(in_path, ".");
lastSlash = lastIndexOf(in_path, "/");
basename= substring(in_path, lastSlash+1, dotPos);   
// print("Basename:" + basename);

// get file extension
extension= substring(in_path, lengthOf(in_path)-3,lengthOf(in_path));
// print("Extension:" + extension);

// get parent folder name
parentPath = substring(in_path, 0, lastSlash);
// print("Parent Path: " + parentPath);

// check for file extension and perform conversion
if(extension=="tif")
	{
	print("Processing image: " + basename);
	open(in_path);
	run("Analyze... ", "save="+out_path +"/"+ basename + ".img");
	print("Saved image: " + basename);
	}
	else
	{
	print("Image "+ basename + " is not in required input format (.tif)");
	}
run("Close All");
run("Quit");
