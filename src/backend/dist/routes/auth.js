import { data } from "../shared/data.js";
import jwt from "jsonwebtoken";
import { service } from "../shared/couchdb_service.js";
export const auth = (req, res) => {
    let username = req.body.username;
    const password = req.body.password;
    let appId = req.body.app_id;
    // **REMOVE
    if (!appId) {
        appId = 1;
    }
    const users = data.users;
    username = username.toLowerCase().trim();
    let response;
    if (username in users) {
        if (users[username].password == password.trim()) {
            if (!users[username].app_id.includes(appId)) {
                res.status(401);
                res.send();
                return;
            }
            const token = jwt.sign({ username: username }, data.tokenKey, {
                expiresIn: data.userAuthTimeout, // represented in seconds
            });
            users[username].token = token;
            response = {
                success: true,
                token: token,
                username: username,
            };
            // register the user login in database
            const docId = Date.now();
            const dateTime = new Date().toLocaleString("pt-BR", {
                timeZone: "Brazil/East",
            });
            const doc = {
                _id: "login:" + username + "_" + docId,
                created: dateTime,
                username: username,
                token: token,
                app_id: appId,
                status: "Sucesso",
            };
            try {
                service
                    .postDocument({
                    db: "assistente-nheengatu",
                    document: doc,
                })
                    .then((response) => {
                    console.log("Cloudant response:" + JSON.stringify(response.result));
                });
            }
            catch (err) {
                console.error("Error:", err);
            }
        }
        else {
            const token = jwt.sign({ username: username }, data.tokenKey, {
                expiresIn: data.userAuthTimeout,
            });
            users[username].token = token;
            const docId = Date.now();
            const dateTime = new Date().toLocaleString("pt-BR", {
                timeZone: "Brazil/East",
            });
            const doc = {
                _id: "login:" + username + "_" + docId,
                created: dateTime,
                username: username,
                token: token,
                app_id: appId,
                status: "Senha Inválida",
            };
            service
                .postDocument({
                db: "assistente-nheengatu",
                document: doc,
            })
                .then((response) => {
                console.log("Cloudant response:" + JSON.stringify(response.result));
            });
            res.status(401);
            res.statusMessage =
                "Senha Incorreta. Verifique se digitou corretamente. Atenção a espaços extras e uso de maiúsculas/minúsculas (Caps Lock).";
            res.json({
                message: "Senha Incorreta. Verifique se digitou corretamente. Atenção a espaços extras e uso de maiúsculas/minúsculas (Caps Lock).",
            });
            return;
        }
    }
    else {
        const docId = Date.now();
        const dateTime = new Date().toLocaleString("pt-BR", {
            timeZone: "Brazil/East",
        });
        const doc = {
            _id: "login:" + username + "_" + docId,
            created: dateTime,
            username: username,
            app_id: appId,
            status: "Usuário inválido",
        };
        service
            .postDocument({
            db: "assistente-nheengatu",
            document: doc,
        })
            .then((response) => {
            console.log("Cloudant response:" + JSON.stringify(response.result));
        });
        res.status(401);
        res.statusMessage =
            "Usuário Inválido. Verifique se digitou corretamente. Atenção a espaços extras e uso de maiúsculas/minúsculas (Caps Lock).";
        res.json({
            message: "Usuário Inválido. Verifique se digitou corretamente. Atenção a espaços extras e uso de maiúsculas/minúsculas (Caps Lock).",
        });
        return;
    }
    res.json(response);
};
//# sourceMappingURL=auth.js.map