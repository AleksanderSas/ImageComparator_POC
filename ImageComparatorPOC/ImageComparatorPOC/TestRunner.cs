using Emgu.CV;
using Emgu.CV.Dnn;
using System.Collections.Concurrent;

namespace ImageComparatorPOC
{
    internal class TestRunner
    {
        private ConcurrentDictionary<string, Feature> _features = new ConcurrentDictionary<string, Feature>();

        private const string ZyteFeatureDir = "C:\\Projects\\watches\\Zyte-features\\";
        private const string ResutsDir = "C:\\Projects\\watches\\comparisionResults\\";
        private const string ZyteDir = "C:\\Projects\\watches\\Zyte-production\\";
        private const string SweDir = "C:\\Projects\\watches\\SWE-production-A\\";

        public async Task RunTestWithCache(string brand)
        {
            var dir = $"{SweDir}{brand}";
            var models = new DirectoryInfo(dir).GetDirectories();
            foreach (var model in models)
            {
                await RunTestWithCache(brand, model.Name);
            }
        }

        public async Task RunTestWithCache(string brand, string model)
        {
            var dir = $"{SweDir}{brand}\\{model}";
            var files = new DirectoryInfo(dir).GetFiles();
            foreach (var file in files)
            {
                try
                {
                    Console.WriteLine(file.Name);
                    await RunTestWithCache(file.Name, brand, model);
                }
                catch (Exception ex) 
                {
                    Console.WriteLine($"\nERROR: {file.Name} - {ex}");
                }
            }
        }

        public async Task RunTestWithCache(string filename, string brand, string model)
        {
            var resultDir = $"{ResutsDir}{brand}\\{model}";
            var resultPath = $"{resultDir}\\{filename}.txt";
            if(File.Exists(resultPath))
            {
                Console.WriteLine("Results already exists");
                return;
            }

            var watchFile = $"{SweDir}{brand}\\{model}\\{filename}";
            var watch = Feature.GetFature(CvInvoke.Imread(watchFile), filename);

            var filesNames = GetFilesToCompare(brand, model);
            //For test purpose
            //filesNames.Files = filesNames.Files.Skip(400).ToList();
            var descriptors = await GetFeature(filesNames.Files);
            var results = await Tester.TestAsync(descriptors, watch);

            if(!Directory.Exists(resultDir))
            {
                Directory.CreateDirectory(resultDir);
            }
            using (StreamWriter outputFile = new StreamWriter(resultPath))
            {
                outputFile.WriteLine($"File: {watchFile}");
                outputFile.WriteLine($"Scan_no: {descriptors.Count}");
                outputFile.WriteLine($"Brands: {string.Join(";", filesNames.Brands)}");
                outputFile.WriteLine($"Models: {string.Join(";", filesNames.Models)}");
                foreach (var r in results.Where(x => x.Score < 0.0 && x.Score > -150.0))
                {
                    outputFile.WriteLine($"{r.Image};{r.Score};{r.ScorePerPoint}");
                }
   
            }
        }

        private FilesToProcess GetFilesToCompare(string brand, string model)
        {
            var result = new FilesToProcess();
            var dirs = new DirectoryInfo(ZyteDir).GetDirectories();
            var brandDirs = dirs.Where(x => IsNameSimilar(brand, x.Name)).ToList();
            foreach (var brandDir in brandDirs)
            {
                result.Brands.Add(brandDir.Name);
                dirs = new DirectoryInfo(brandDir.FullName).GetDirectories();
                var modelDirs = dirs.Where(x => IsNameSimilar(model, x.Name)).ToList();

                foreach (var modelDir in modelDirs)
                {
                    result.Models.Add(modelDir.Name);
                    var files = new DirectoryInfo(modelDir.FullName).GetFiles();
                    result.Files.AddRange(files.Select(x => x.FullName));
                }
            }
            return result;
        }

        private async Task<List<Feature>> GetFeature(List<string> filenames)
        {
            var readImageContext = new ParallelContext { TotalCount = filenames.Count };
            var batches = filenames.ToList().Batches(filenames.Count / 10);
            //var batches = filenames.ToList().Batches(filenames.Count);
            var taskResults = await Task.WhenAll(batches.Select(x => GetFeatureAsync(x, readImageContext)).ToList());
            Console.WriteLine();

            List<Feature> descriptors = taskResults
                .SelectMany(x => x)
                .Where(x => x != null)
                .Cast<Feature>()
                .ToList();

            return descriptors;
        }

        Task<List<Feature?>> GetFeatureAsync(IList<string> files, ParallelContext context)
        {
            return Task.Run(() => files.Select(y =>
            {
                try
                {
                    if (!_features.TryGetValue(y, out var feature))
                    {
                        feature = ReadFeature(y);
                    }

                    lock (context)
                    {
                        Console.Write($"\rRead images {++context.FinishedCount}\\{context.TotalCount}");
                    }
                    return feature;
                }
                catch(Exception ex) 
                {
                    return null;
                }
            }).ToList());
        }

        private Feature ReadFeature(string filename)
        {
            var cachFile = filename.Replace(ZyteDir, ZyteFeatureDir) + ".PFM";
            var feature = Feature.Load(cachFile, filename);
            if(feature == null)
            {
                feature = Feature.GetFature(CvInvoke.Imread(filename), filename);
                if (feature != null)
                {
                    feature.Save(cachFile);
                }
            }

            
            if (feature != null)
            {
                _features[filename] = feature;
            }

            return feature;
        }

        private bool IsNameSimilar(string swe, string zyte)
        {
            swe = swe.Replace("'", "").Replace("\"", " ").Replace("&", " ").Replace("-", " ");
            var weWords = swe.Split(' ').Where(x => x.Length > 2).ToList();

            zyte = zyte.Replace("'", "").Replace("\"", " ").Replace("&", " ").Replace("-", " ");
            var zyteWords = zyte.Split(' ').Where(x => x.Length > 2).ToList();

            if(weWords.Count == 0|| zyteWords.Count == 0)
            {
                //Console.WriteLine($"'{s1}' '{s2}' fallback");
                return true;
            }

            return weWords.All(zyteWords.Contains);
        }

        struct FilesToProcess
        {
            public List<string> Files;
            public List<string> Brands;
            public List<string> Models;

            public FilesToProcess()
            {
                Files = new List<string>();
                Brands= new List<string>();
                Models= new List<string>();
            }
        }
    }
}
