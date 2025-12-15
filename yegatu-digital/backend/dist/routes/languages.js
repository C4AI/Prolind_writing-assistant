import { data } from "../shared/data.js";
export const languages = (req, res) => {
    const username = req.body.username;
    try {
        const response = {
            languages: data.languages,
            language: data.users[username].language,
            orthography: data.users[username].orthography,
            enable_model_change: data.enableModelChange,
        };
        res.status(200);
        res.json(response);
    }
    catch (err) {
        console.error("Error:", err);
    }
};
//# sourceMappingURL=languages.js.map