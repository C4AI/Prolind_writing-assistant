import { CloudantV1, IamAuthenticator } from "@ibm-cloud/cloudant";
const authenticator = new IamAuthenticator({
    apikey: `${process.env.CLOUDANT_API_KEY}`
});
export const service = new CloudantV1({
    authenticator: authenticator
});
service.setServiceUrl("https://apikey-v2-3grjp7w7jyq19amlcmt1b55p0b9n0zcgc0ur5yrobj2:a5a2c56139a67fd241da81e000b4a94e@8abf66c6-46c9-4fba-8abe-213ecd59573b-bluemix.cloudantnosqldb.appdomain.cloud");
//# sourceMappingURL=cloudante_service.js.map