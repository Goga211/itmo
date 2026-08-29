package ru.traphouse.functions.log;

public class Log2 {
    private final Ln ln;
    private final double LN2;

    public Log2(Ln ln) {
        this.ln = ln;
        this.LN2 = ln.calculate(2.0);
    }

    // log₂(x) = ln(x) / ln(2)
    public double calculate(double x) {
        return ln.calculate(x) / LN2;
    }
}
