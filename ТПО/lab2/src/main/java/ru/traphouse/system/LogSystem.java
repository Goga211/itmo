package ru.traphouse.system;

import ru.traphouse.functions.log.Ln;
import ru.traphouse.functions.log.Log10;
import ru.traphouse.functions.log.Log2;
import ru.traphouse.functions.log.Log3;
import ru.traphouse.functions.log.Log5;

/**
 * Логарифмическая часть системы (x > 0):
 * (((((log_10(x) + ln(x)) + log_10(x)) - (log_3(x) ^ 3)) * log_10(x))
 *  / ((log_3(x) + ((log_5(x) - log_2(x)) * log_2(x))) / ((log_10(x) - log_10(x)) - ln(x))))
 */
public class LogSystem {
    private final Ln ln;
    private final Log2 log2;
    private final Log3 log3;
    private final Log5 log5;
    private final Log10 log10;

    public LogSystem(Ln ln, Log2 log2, Log3 log3, Log5 log5, Log10 log10) {
        this.ln = ln;
        this.log2 = log2;
        this.log3 = log3;
        this.log5 = log5;
        this.log10 = log10;
    }

    public double calculate(double x) {
        double lnVal    = ln.calculate(x);
        double log2Val  = log2.calculate(x);
        double log3Val  = log3.calculate(x);
        double log5Val  = log5.calculate(x);
        double log10Val = log10.calculate(x);

        double numerator = ((log10Val + lnVal + log10Val) - (log3Val * log3Val * log3Val)) * log10Val;

        double denomTop = log3Val + (log5Val - log2Val) * log2Val;
        double denomBot = (log10Val - log10Val) - lnVal; // = -ln(x)

        // x = 1: ln(x) = 0 → denomBot = 0 → явная точка разрыва функции
        if (Math.abs(denomBot) < 1e-10) {
            throw new ArithmeticException("LogSystem undefined at x = 1 (ln(x) = 0)");
        }

        // denomTop = 0 → результат ±Infinity (поведение double корректно)
        return numerator / (denomTop / denomBot);
    }
}
