package ru.traphouse.stubs.log;

import ru.traphouse.functions.log.Ln;

/**
 * Заглушка Ln с фиксированным возвращаемым значением.
 * Используется для изолированного тестирования граничных условий в LogSystem.
 */
public class FixedLn extends Ln {
    private final double fixedValue;

    public FixedLn(double fixedValue) {
        super(1e-10);
        this.fixedValue = fixedValue;
    }

    @Override
    public double calculate(double x) {
        return fixedValue;
    }
}
