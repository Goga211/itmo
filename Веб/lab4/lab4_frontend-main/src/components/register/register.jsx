import {Button} from "baseui/button";
import CONFIG from "../../config";
import axiosInstance from  "../../axiosInstance"

import {
    CenteredHeadingXXLarge,
    Container,
    ErrorText,
    Head,
    InnerContainer,
    InputWrapper,
    StyledInput
} from "../commons";
import {validationSchema} from "../validationSchema";

import {useFormik} from "formik";

import axios, {AxiosError} from "axios";
import {useState} from "react";
import {useNavigate} from "react-router-dom";

function Register() {
    const [error, setError] = useState("");
    const navigate = useNavigate();

    const handleRedirect = () => {
        navigate("/login");
    };


    const onSubmit = async (values) => {
        console.log("Values: ", values);
        setError(""); // Сбрасываем предыдущую ошибку

        try {
            await axiosInstance.post("/auth/signup", values);
            navigate("/login"); // Перенаправление при успешной регистрации
        } catch (err) {
            if (err && err instanceof AxiosError) {
                // Извлекаем сообщение об ошибке из ответа сервера
                const serverMessage = err.response?.data.error || "An unexpected error occurred.";
                setError(serverMessage); // Сохраняем сообщение об ошибке
            } else if (err && err instanceof Error) {
                setError(err.message);
            }
            console.log("Error: ", err);
        }
    };


    const formik = useFormik({
        initialValues: {
            username: "",
            password: "",
        },
        validationSchema, // Подключаем схему валидации
        onSubmit,
    });

    return (
        <>
            <Head>Антипин Григорий Викторович, P3332, 2223</Head>
            <Container>

                <InnerContainer>
                    <form onSubmit={formik.handleSubmit}>
                        <CenteredHeadingXXLarge>Let's register</CenteredHeadingXXLarge>

                        <ErrorText>{error}</ErrorText>
                        <InputWrapper>
                            <div style={{display: "flex", flexDirection: "column", width: "100%"}}>
                                <StyledInput
                                    name="username"
                                    value={formik.values.username}
                                    onChange={formik.handleChange}
                                    onBlur={formik.handleBlur} // Отслеживание фокуса
                                    placeholder="Username"
                                    clearOnEscape
                                    size="large"
                                    type="text"
                                />
                                {formik.touched.username && formik.errors.username && (
                                    <ErrorText>{formik.errors.username}</ErrorText>
                                )} </div>
                        </InputWrapper>
                        <InputWrapper>
                            <div style={{display: "flex", flexDirection: "column", width: "100%"}}>
                                <StyledInput
                                    name="password"
                                    value={formik.values.password}
                                    onChange={formik.handleChange}
                                    onBlur={formik.handleBlur} // Отслеживание фокуса
                                    placeholder="Password"
                                    clearOnEscape
                                    size="large"
                                    type="password"
                                />
                                {formik.touched.password && formik.errors.password && (
                                    <ErrorText>{formik.errors.password}</ErrorText>
                                )} </div>
                        </InputWrapper>
                        <InputWrapper>
                            <Button type="submit" isLoading={formik.isSubmitting}>
                                Register
                            </Button>
                            <Button onClick={handleRedirect}>
                                Go login
                            </Button>
                        </InputWrapper>
                    </form>
                </InnerContainer>
            </Container>
        </>
    );
}

export {Register};
