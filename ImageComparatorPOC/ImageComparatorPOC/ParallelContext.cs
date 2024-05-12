namespace ImageComparatorPOC;

internal class ParallelContext
{
    public ParallelContext()
    {
        StartTime = DateTime.UtcNow;
    }

    public int TotalCount { get; set; }

    public int FinishedCount;

    public int ThreadCount;

    public DateTime StartTime { get; set; }

    public void Decrement()
    {
        Interlocked.Decrement(ref ThreadCount);
    }

    public int MilisPerImage()
    {
        if(FinishedCount <= 0)
            return 0;
        return (int)((DateTime.UtcNow - StartTime).TotalMilliseconds / FinishedCount);
    }
}
