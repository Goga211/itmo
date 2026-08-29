package ru.traphouse.stubs.log;

import ru.traphouse.functions.log.Log3;

/**
 * Заглушка Log3 с фиксированным возвращаемым значением.
 */
public class FixedLog3 extends Log3 {
    private final double fixedValue;

    public FixedLog3(double fixedValue) {
        super(new FixedLn(1.0));
        this.fixedValue = fixedValue;
    }

    @Override
    public double calculate(double x) {
        return fixedValue;
    }
}
