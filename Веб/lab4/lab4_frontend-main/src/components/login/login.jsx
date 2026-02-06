import {Button} from "baseui/button";
import {HeadingXXLarge,} from "baseui/typography";
import {Container, ErrorText, Head, InnerContainer, InputWrapper, StyledInput} from "../commons";
import {validationSchema} from "../validationSchema";
import {useFormik} from "formik";
import axios, {AxiosError} from "axios";
import {useState} from "react";
import {useNavigate} from "react-router-dom";
import CONFIG from "../../config";
import axiosInstance from  "../../axiosInstance"

function Login() {
    const [error, setError] = useState("");

    const navigate = useNavigate();


    const handleRedirect = () => {
        navigate("/register")
    }


    const onSubmit = async (values) => {
        console.log("Values: ", values);
        setError("");


        try {
            const response = await axiosInstance.post(
                "/auth/signin",
                values
            );
            console.log(response.status)


            const username = response.data.username
            const accessToken = response.data.accessToken
            const refreshToken = response.data.refreshTokenHash


            localStorage.setItem("username", username)
            localStorage.setItem("accessToken", accessToken)
            localStorage.setItem("refreshToken", refreshToken)

            console.log(refreshToken)
            console.log(accessToken)


            navigate("/home")

        } catch (err) {

            if (err && err instanceof AxiosError)
                setError(err.response?.data.error);
            else if (err && err instanceof Error) setError(err.message);

            console.log("Error: ", err);
        }
    };

    const formik = useFormik({
        initialValues: {
            username: "",
            password: "",
        },
        validationSchema,
        onSubmit,
    });

    return (
        <><Head>Антипин Григорий Викторович, P3332, 2223</Head><Container>
            <InnerContainer>
                <form onSubmit={formik.handleSubmit}>
                    <HeadingXXLarge>Welcome Back!</HeadingXXLarge>
                    <ErrorText>{error}</ErrorText>
                    <InputWrapper>
                        <div style={{display: "flex", flexDirection: "column", width: "100%"}}>
                            <StyledInput
                                name="username"
                                value={formik.values.username}
                                onChange={formik.handleChange}
                                onBlur={formik.handleBlur}
                                placeholder="Username"
                                clearOnEscape
                                size="large"
                                type="text"/>
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
                                onBlur={formik.handleBlur}
                                placeholder="Password"
                                clearOnEscape
                                size="large"
                                type="password"/>
                            {formik.touched.password && formik.errors.password && (
                                <ErrorText>{formik.errors.password}</ErrorText>
                            )} </div>

                    </InputWrapper>
                    <InputWrapper>
                        <Button type="submit" isLoading={formik.isSubmitting}>
                            Login
                        </Button>
                        <Button onClick={handleRedirect}>
                            Go register
                        </Button>
                    </InputWrapper>

                </form>
            </InnerContainer>
        </Container></>

    );
}

export {Login};
