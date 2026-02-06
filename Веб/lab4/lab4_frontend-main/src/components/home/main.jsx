import styled from "styled-components";
import {Button} from "baseui/button";
import React, {useEffect} from "react";
import CoordinatePlate from "./plate";
import CoordinateForm from "./form";
import Table from "./table";
import {useDispatch, useSelector} from "react-redux";
import {editName} from "../../redux/usernameSlice";
import {logout as performLogout} from "../../axiosInstance";
import 'primereact/resources/themes/lara-light-blue/theme.css';
import 'primereact/resources/primereact.min.css';
import 'primeicons/primeicons.css';
import 'primeflex/primeflex.css';

// Контейнер для всей страницы
export const Container = styled.div`
    width: 100%;
    min-height: 100vh; /* Растягивает контейнер на всю высоту экрана */
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    background-color: #1c1c1c;
    user-select: none;
    box-sizing: border-box;

    @media (max-width: 891px) {
        padding: 1rem; /* Отступы для мобильных устройств */
    }
`;



const HeaderContainer = styled.div`
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    display: flex;
    justify-content: space-between; /* Разместить элементы по краям */
    align-items: center;
    background-color: #1c1c1c;
    padding: 10px 20px;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    
    
    @media (max-width: 891px) {
        flex-direction: column; /* Размещаем элементы в колонку */
        align-items: flex-start; /* Выровнять по левому краю */
        padding: 10px;
    }
`;


const Greeting = styled.h1`
    color: white;
    margin: 0;
    font-size: 18px;

    @media (max-width: 891px) {
        font-size: 16px; /* Уменьшаем шрифт для мобильных */
        margin-bottom: 10px; /* Добавляем отступ снизу */
    }
`;


const LogoutButton = styled(Button)`
    border-radius: 20px;
    margin-right: 30px;

    :hover {
        background-color: red;
    }

    @media (max-width: 891px) {
        margin-right: 0; /* Убираем отступ справа */
        width: 100%; /* Растягиваем кнопку на всю ширину */
        margin-top: 10px; /* Отступ сверху */
    }
`;



const MainContainer = styled.div`
    display: flex;
    flex-grow: 1;
    align-items: flex-start;
    gap: 20px;
    width: 100%;
    margin-top: 70px; /* Отступ равен высоте HeaderContainer */
    box-sizing: border-box;
    
    

    @media (max-width: 1259px) {
        flex-direction: column; /* Переводим элементы в колонку для планшетов */
        align-items: center; /* Центрируем элементы */
        gap: 10px; /* Уменьшаем расстояние между элементами */
    }

    @media (max-width: 891px) {
        margin-top: 90px; /* Увеличиваем отступ для мобильных */
    }
`;



function Home() {
    const name = useSelector((state) => state.usernameEditor.username);
    const dispatch = useDispatch();

    useEffect(() => {
        // Если имя пользователя отсутствует в Redux, восстановите его из localStorage
        const storedName = localStorage.getItem("username");
        if (storedName) {
            dispatch(editName(storedName));
        }
    }, [dispatch]);

    const handleLogout = () => {
        performLogout(dispatch); // Передаём dispatch в функцию logout
    };


    return (
        <Container>
            <HeaderContainer>
                <Greeting>Hello, {name}</Greeting>
                <LogoutButton onClick={handleLogout}>Logout</LogoutButton>
            </HeaderContainer>

            <MainContainer>
                <CoordinatePlate/>
                <CoordinateForm/>
                <Table/>
            </MainContainer>
        </Container>
    );
}

export {Home};
