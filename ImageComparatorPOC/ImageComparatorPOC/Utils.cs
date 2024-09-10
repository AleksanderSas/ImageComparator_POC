namespace ImageComparatorPOC;

internal static class Utils
{
    public static List<IList<T>> Batches<T>(this IList<T> input, int elementsPerBatch)
    {
        if(elementsPerBatch == 0)
        {
            return new List<IList<T>> { input };
        }
        int k = 0;
        List<IList<T>> result = new List<IList<T>> ();
        while (k < input.Count) 
        {
            var tmp = new List<T>();
            for(int m = 0; m < elementsPerBatch && m + k < input.Count; m++) 
            {
                tmp.Add(input[k + m]);
            }
            k += elementsPerBatch;
            result.Add(tmp);// yield return tmp;
        }
        return result;
    }
}
