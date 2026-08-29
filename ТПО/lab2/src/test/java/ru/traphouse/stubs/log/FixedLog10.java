package ru.traphouse.stubs.log;

import ru.traphouse.functions.log.Log10;

/**
 * Заглушка Log10 с фиксированным возвращаемым значением.
 */
public class FixedLog10 extends Log10 {
    private final double fixedValue;

    public FixedLog10(double fixedValue) {
        super(new FixedLn(1.0));
        this.fixedValue = fixedValue;
    }

    @Override
    public double calculate(double x) {
        return fixedValue;
    }
}
