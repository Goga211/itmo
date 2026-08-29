package ru.traphouse.functions.log;

public class Log10 {
    private final Ln ln;
    private final double LN10;

    public Log10(Ln ln) {
        this.ln = ln;
        this.LN10 = ln.calculate(10.0);
    }

    // log₁₀(x) = ln(x) / ln(10)
    public double calculate(double x) {
        return ln.calculate(x) / LN10;
    }
}
