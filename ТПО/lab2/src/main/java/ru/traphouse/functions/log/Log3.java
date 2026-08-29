package ru.traphouse.functions.log;

public class Log3 {
    private final Ln ln;
    private final double LN3;

    public Log3(Ln ln) {
        this.ln = ln;
        this.LN3 = ln.calculate(3.0);
    }

    // log₃(x) = ln(x) / ln(3)
    public double calculate(double x) {
        return ln.calculate(x) / LN3;
    }
}
