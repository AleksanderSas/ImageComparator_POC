using Emgu.CV;
using System.Collections.Concurrent;

namespace ImageComparatorPOC
{
    internal class TestRunner
    {
        private ConcurrentDictionary<string, Feature> _features = new ConcurrentDictionary<string, Feature>();

        private const string ZyteFeatureDir = "C:\\Projects\\watches\\Zyte-features\\";
        private const string ResultsDir = "C:\\Projects\\watches\\comparisionResults-S\\";
        private const string ResultsDir2 = "C:\\Projects\\watches\\comparisionResults\\SWE_SINGLE\\";
        private const string ZyteDir = "C:\\Projects\\watches\\Zyte-production\\";
        private const string SweDir = "C:\\Projects\\watches\\SWE-production-S\\";

        public async Task ScanSwe(string image1, string image2, string brand, string model, bool useGeometryFeature)
        {
            var dir = $"{SweDir}{brand}\\{model}";
            var models = new DirectoryInfo(dir).GetFiles();
            List<Feature> descriptors = await GetFeature(models.Select(x => x.FullName).ToList(), false, useGeometryFeature);
            await ScanSwe2(image1, brand, model, descriptors);
            await ScanSwe2(image2, brand, model, descriptors);
        }

        public async Task RunTestWithCache(string brand, bool useGeometryFeature)
        {
            var dir = $"{SweDir}{brand}";
            var models = new DirectoryInfo(dir).GetDirectories();
            foreach (var model in models)
            {
                await RunTestWithCache(brand, model.Name, useGeometryFeature);
            }
        }

        public async Task RunTestWithCache(string brand, string model, bool useGeometryFeature)
        {
            var dir = $"{SweDir}{brand}\\{model}";
            var files = new DirectoryInfo(dir).GetFiles();
            foreach (var file in files)
            {
                try
                {
                    Console.WriteLine(file.Name);
                    await RunTestWithCache(file.Name, brand, model, useGeometryFeature);
                }
                catch (Exception ex) 
                {
                    Console.WriteLine($"\nERROR: {file.Name} - {ex}");
                }
            }
        }

        public async Task ScanSwe2(string filename, string brand, string model, List<Feature> descriptors)
        {
            var resultDir = $"{ResultsDir2}{brand}\\{model}";

            var idx = Math.Max(filename.LastIndexOf('/'), filename.LastIndexOf('\\'));
            var name = filename.Substring(idx + 1); 
            var watch = Feature.GetFature(CvInvoke.Imread(filename), name);
            var resultPath = $"{resultDir}\\{name}.txt";

            //For test purpose
            //filesNames.Files = filesNames.Files.Skip(400).ToList();
            var results = await Tester.TestAsync(descriptors, watch, true);

            if (!Directory.Exists(resultDir))
            {
                Directory.CreateDirectory(resultDir);
            }
            using (StreamWriter outputFile = new StreamWriter(resultPath))
            {
                outputFile.WriteLine($"File: {filename}");
                outputFile.WriteLine($"Scan_no: {descriptors.Count}");
                outputFile.WriteLine($"Brands: {brand}");
                outputFile.WriteLine($"Models: {model}");
                foreach (var r in results.Where(x => x.Score < 0.0))// && x.Score > -180.0))
                {
                    outputFile.WriteLine($"{r.Image};{r.Score};{r.ScorePerPoint}");
                }

            }
        }

        public async Task RunTestWithCache(string filename, string brand, string model, bool useGeometryFeature)
        {
            var resultDir = $"{ResultsDir}{brand}\\{model}";
            var resultPath = $"{resultDir}\\{filename}.txt";
            if (File.Exists(resultPath))
            {
                Console.WriteLine("Results already exists");
                return;
            }

            var watchFile = $"{SweDir}{brand}\\{model}\\{filename}";
            var watch = Feature.GetFature(CvInvoke.Imread(watchFile), filename);

            var filesNames = GetFilesToCompare(brand, model);
            //For test purpose
            //filesNames.Files = filesNames.Files.Skip(400).ToList();
            await RunTestWithCache(useGeometryFeature, resultDir, resultPath, watchFile, watch, filesNames);
        }

        public async Task RunTestWithCache(bool useGeometryFeature, string resultDir, string resultPath, string watchFile, Feature watch, FilesToProcess filesNames)
        {
            var descriptors = await GetFeature(filesNames.Files, true, useGeometryFeature);
            var results = await Tester.TestAsync(descriptors, watch, useGeometryFeature);

            if (!Directory.Exists(resultDir))
            {
                Directory.CreateDirectory(resultDir);
            }
            using (StreamWriter outputFile = new StreamWriter(resultPath))
            {
                var version = useGeometryFeature ? 1 : 0;
                outputFile.WriteLine($"Version: {useGeometryFeature}");
                outputFile.WriteLine($"File: {watchFile}");
                outputFile.WriteLine($"Scan_no: {descriptors.Count}");
                outputFile.WriteLine($"Brands: {string.Join(";", filesNames.Brands)}");
                outputFile.WriteLine($"Models: {string.Join(";", filesNames.Models)}");
                var scoreLimit = useGeometryFeature ? -250.0 : -180.0;
                foreach (var r in results.Where(x => x.Score < 0.0 && x.Score > scoreLimit))
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

        private async Task<List<Feature>> GetFeature(List<string> filenames, bool saveFeatures, bool useGeometryFeature)
        {
            var readImageContext = new ParallelContext { TotalCount = filenames.Count };
            var batches = filenames.ToList().Batches(filenames.Count / 6);
            //var batches = filenames.ToList().Batches(filenames.Count);
            var taskResults = await Task.WhenAll(batches.Select(x => GetFeatureAsync(x, readImageContext, saveFeatures, useGeometryFeature)).ToList());
            Console.WriteLine();

            List<Feature> descriptors = taskResults
                .SelectMany(x => x)
                .Where(x => x != null)
                .Cast<Feature>()
                .ToList();

            return descriptors;
        }

        Task<List<Feature?>> GetFeatureAsync(IList<string> files, ParallelContext context, bool saveFeatures, bool useGeometryFeature)
        {
            return Task.Run(() => files.Select(y =>
            {
                try
                {
                    if (!_features.TryGetValue(y, out var feature))
                    {
                        feature = ReadFeature(y, saveFeatures, useGeometryFeature);
                    }

                    var c = Interlocked.Increment(ref context.FinishedCount);
                    if(c % 8 == 0)
                    {
                        lock (context)
                        {
                            Console.Write($"\rRead images {c}\\{context.TotalCount} MPI: {context.MilisPerImage()}");

                        }
                    }
                    return feature;
                }
                catch(Exception ex) 
                {
                    return null;
                }
            }).ToList());
        }

        private Feature ReadFeature(string filename, bool saveFeatures, bool useGeometryFeature)
        {
            var cachFile = filename.Replace(ZyteDir, ZyteFeatureDir) + ".PFM";
            var feature = Feature.Load(cachFile, filename, useGeometryFeature);
            if(feature == null)
            {
                feature = Feature.GetFature(CvInvoke.Imread(filename), filename);
                if (saveFeatures && feature != null)
                {
                    feature.Save(cachFile);
                }
            }

                _features[filename] = feature;

            return feature;
        }

        private bool IsNameSimilar(string swe, string zyte)
        {
            swe = swe.Replace("'", "").Replace("\"", " ").Replace("&", " ").Replace("-", " ");
            var weWords = swe.Split(' ').Where(x => x.Length > 1).ToList();

            zyte = zyte.Replace("'", "").Replace("\"", " ").Replace("&", " ").Replace("-", " ");
            var zyteWords = zyte.Split(' ').Where(x => x.Length > 1).ToList();

            if(weWords.Count == 0|| zyteWords.Count == 0)
            {
                //Console.WriteLine($"'{s1}' '{s2}' fallback");
                return true;
            }

            return weWords.All(zyteWords.Contains);
        }

        public struct FilesToProcess
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
