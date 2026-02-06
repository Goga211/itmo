import { Input } from "baseui/input";
import styled from "styled-components";
import {HeadingXXLarge} from "baseui/typography";

// Контейнер для всей страницы
export const Container = styled.div`
    width: 100%;
    min-height: 100vh; /* Минимальная высота на весь экран */
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    background-color: #1c1c1c;
    box-sizing: border-box;
    user-select: none;


    @media (max-width: 891px) {
        padding: 1rem; /* Уменьшение отступов на мобильных */
    }
`;



export const CenteredHeadingXXLarge = styled(HeadingXXLarge)`
    text-align: center;
    margin-bottom: 1rem;
`;



// Внутренний контейнер для формы
export const InnerContainer = styled.div`
    margin: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 2rem;
    border-radius: 1rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    background-color: #1c1c1c;
    width: 100%;
    max-width: 400px;

    @media (max-width: 1259px) {
        max-width: 300px; /* Уменьшение ширины для планшетов */
    }

    @media (max-width: 891px) {
        max-width: 90%; /* Почти на всю ширину экрана на мобильных */
        padding: 2rem 1.5rem;
    }
`;


// Контейнер для полей ввода
export const InputWrapper = styled.div`
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    margin: 1rem 0;

    button {
        border-radius: 20px;
        margin: 0 0.5rem;
    }

    button:hover {
        background-color: aquamarine;
    }

    @media (max-width: 1259px) {
        margin: 0.8rem 0; /* Уменьшение отступов для планшетов */
    }

    @media (max-width: 891px) {
        margin: 0.5rem 0; /* Еще меньше отступов для мобильных */
        flex-direction: column; /* Кнопки и поля ввода в колонку */
        button {
            margin: 0.5rem 0; /* Вертикальный отступ между кнопками */
        }
    }
`;



// Стили для поля ввода
export const StyledInput = styled(Input)`
    width: 100%;
    font-size: 16px;
    padding: 0.8rem;

    @media (max-width: 1259px) {
        font-size: 14px; /* Уменьшение шрифта для планшетов */
    }

    @media (max-width: 891px) {
        font-size: 12px; /* Уменьшение шрифта для мобильных */
        padding: 0.6rem; /* Меньше отступов */
    }
`;


// Стили для текста ошибок
export const ErrorText = styled.span`
    color: #eb5d5d;
    font-size: 14px;
    margin-top: 5px; /* Отступ сверху */
    width: 100%; /* Полностью под ширину поля */
    text-align: left; /* Выравнивание ошибки по левому краю */
`;


// Заголовок страницы
export const Head = styled.div`
    width: 100%;
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    color: #ffffff;
    text-align: center;
    background-color: #1c1c1c;
    font-size: 40px;
    font-weight: bold;
    padding: 1rem 0;
    user-select: none;

    @media (max-width: 1259px) {
        font-size: 32px; /* Уменьшение размера шрифта на планшетах */
    }

    @media (max-width: 891px) {
        font-size: 24px; /* Уменьшение размера шрифта на мобильных */
    }
`;


