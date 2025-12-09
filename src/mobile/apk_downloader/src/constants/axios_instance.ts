import axios from "axios";

// create an instance of axios with the base URL of the API
export const axiosInstance = axios.create({
    baseURL: import.meta.env.VITE_API_URL
});

export const axiosTranslator = axios.create({
    baseURL: import.meta.env.VITE_API_URL
});

export const axiosCorrector = axios.create({
    baseURL: import.meta.env.VITE_API_URL
});

// intercept the requests of axios and add the token from the localstorage
axiosInstance.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem("token");

        if (token) {
            config.headers["Authorization"] = `${token}`;
        }

        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// intercept the responses of axios and redirect to the login page if the status code is 401 (unauthorized)
axiosInstance.interceptors.response.use(
    (response) => {
        return response;
    },
    (error) => {
        if (error.response && error.response.status === 401) {
            if (window.location.href.endsWith("en")) {
                window.location.href = "/en";
            } else {
                window.location.href = "/";
            }
        }
        return Promise.reject(error);
    }
);

// intercept the requests of axios and add the token from the localstorage
axiosTranslator.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem("token");

        if (token) {
            config.headers["Authorization"] = `${token}`;
        }

        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// intercept the responses of axios and redirect to the login page if the status code is 401 (unauthorized)
axiosTranslator.interceptors.response.use(
    (response) => {
        return response;
    },
    (error) => {
        if (error.response && error.response.status === 401) {
            if (window.location.href.endsWith("en")) {
                window.location.href = "/en";
            } else {
                window.location.href = "/";
            }
        }
        return Promise.reject(error);
    }
);

// intercept the requests of axios and add the token from the localstorage
axiosCorrector.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem("token");

        if (token) {
            config.headers["Authorization"] = `${token}`;
        }

        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// intercept the responses of axios and redirect to the login page if the status code is 401 (unauthorized)
axiosCorrector.interceptors.response.use(
    (response) => {
        return response;
    },
    (error) => {
        if (error.response && error.response.status === 401) {
            if (window.location.href.endsWith("en")) {
                window.location.href = "/en";
            } else {
                window.location.href = "/";
            }
        }
        return Promise.reject(error);
    }
);
