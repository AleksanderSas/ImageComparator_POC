namespace ImageComparatorPOC;

internal class ComparisionResult : IComparable<ComparisionResult>
{
    public ComparisionResult(double score, string image)
    {
        Score = score;
        Image = image;
    }

    public double Score { get; set; }

    public string Image {  get; set; }

    public int CompareTo(ComparisionResult? other) => other.Score.CompareTo(Score);
}
