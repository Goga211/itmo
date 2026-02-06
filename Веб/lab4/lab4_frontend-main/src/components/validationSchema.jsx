import * as Yup from "yup";

export const validationSchema = Yup.object({
    username: Yup.string()
        .required("Username is required"), // Поле обязательно
    password: Yup.string()
        .min(8, "Password must be at least 8 characters long") // Минимум 8 символов
        .matches(/^\S*$/, "Password cannot contain spaces") // Запрет пробелов
        .required("Password is required"), // Поле обязательно
});
