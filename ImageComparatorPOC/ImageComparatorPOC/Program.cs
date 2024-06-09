// See https://aka.ms/new-console-template for more information
using Emgu.CV;
using ImageComparatorPOC;
using System.Diagnostics;


//await new TestRunner().ScanSwe("C:\\Projects\\watches\\Single\\Jaeger-LeCoultre Reverso Duoface.jpg", "C:\\Projects\\watches\\Single\\Jaeger-LeCoultre Reverso Duoface", "Jaeger LeCoultre", "Reverso");

//"a-lange-and-sohne-grand-lange-1-rose-gold-mens-watch-115032-51092_267bc.jpg", "Lange 1"

//await new TestRunner().RunTestWithCache("Omega", false);
await (new TestRunner().RunTestWithCache("cartier-tank-louis-small-yellow-gold-brown-strap-ladies-watch-w1529856-60139_a9cfa.jpg",
    "Cartier", "Tank Louis", true));
//RunTest();
//await SearchInDirectory();

static async Task SearchInDirectory()
{
    //Patek Philippe
    string directory = "C:\\Projects\\watches\\SWE-production\\Patek Philippe";
    //string directory = "C:\\Projects\\watches\\SWE-production\\Vacheron Constantin";
    string[] files = Directory.GetFiles(directory);//.Take(10).ToArray();

    //Perek_philippe
    //string teseed2 = "C:\\Projects\\watches\\SWE-production\\Stolen\\Vacheron_constatin_Overseas.jpg";
    string teseed2 = "C:\\Projects\\watches\\SWE-production\\Stolen\\Perek_philippe.jpg";
    var testedDescriptor2 = Feature.GetFature(CvInvoke.Imread(teseed2), "test.jpg");

    //For quick test
    //files = files.Take(40).ToArray();

    var timer = new Stopwatch();
    timer.Start();
    var readImageContext = new ParallelContext { TotalCount = files.Length };
    var batches = files.ToList().Batches(files.Length / 10);
    var taskResults = await Task.WhenAll(batches.Select(x => GetFeatureAsync(x, readImageContext)).ToList());

    List<Feature> descriptors = taskResults
        .SelectMany(x => x)
        .Where(x => x != null)
        .ToList();

    Console.WriteLine($"\nRead time: {timer.ElapsedMilliseconds / 1000}");
    Console.WriteLine();

    timer.Restart();
    var results = await Tester.TestAsync(descriptors, testedDescriptor2, false);
    timer.Stop();
    Console.WriteLine($"\nProcess time: {timer.ElapsedMilliseconds / 1000}");

    var resultPath = $"C:\\Projects\\watches\\test_new_alg.txt";
    using (StreamWriter outputFile = new StreamWriter(resultPath))
    {
        Console.WriteLine();
        foreach (var r in results)
        {
            //Console.WriteLine($"{r.Image} {r.Score}");
            outputFile.WriteLine($"{r.Image} {r.Score}   {r.DiffScore}   {r.AngleScore}");
        }
    }
}

static Task<List<Feature>> GetFeatureAsync(IList<string> files, ParallelContext context)
{
    return Task.Run(() => files.Select(y =>
    {
        var tmp = Feature.GetFature(CvInvoke.Imread(y), y);
        lock(context)
        {
            Console.Write($"\rRead images {++context.FinishedCount}\\{context.TotalCount}");
        }
        return tmp;
    }).ToList());
}