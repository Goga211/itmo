package ru.goga.lab2;

import java.util.function.Predicate;

public class Validator {
    public void validate(float x, int y, int r) throws ValidateException {
        validateValue(x, value -> value >= -2 && value <= 2, "Неверное значение X");
        validateValue(y, value -> value >= -5 && value <= 3, "Неверное значение Y");
        validateValue(r, value -> value >= 1 && value <= 5, "Неверное значение R");
    }

    private <T extends Number> void validateValue(T value, Predicate<T> condition, String errorMessage) throws ValidateException {
        try {
            if (!condition.test(value)) {
                throw new ValidateException(errorMessage);
            }
        } catch (NumberFormatException e) {
            throw new ValidateException(value + " не является номером");
        }
    }
}


