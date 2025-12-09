//import * as Nano  from 'nano';
import Nano from 'nano';
class CouchDBService {
    constructor(couchdbUrl) {
        this.couchdbUrl = couchdbUrl;
    }
    async getDocument(getDocumentParams) {
        try {
            const service = Nano(this.couchdbUrl);
            const db = service.use(getDocumentParams.db);
            const doc = await db.get(getDocumentParams.docId);
            const response = { 'result': doc };
            return response;
        }
        catch (error) {
            console.error(`couchdb_service.getDocument(): Error fetching document ${getDocumentParams.docId} from ${getDocumentParams.db}:`, error);
            throw error; // Re-throw the error for handling by the caller
        }
    }
    async postDocument(postDocumentParams) {
        try {
            const service = Nano(this.couchdbUrl);
            const db = service.use(postDocumentParams.db);
            // Insert the document into the database
            const resp = await db.insert(postDocumentParams.document);
            // resp will contain the _id and _rev of the newly created document
            const response = { 'result': resp };
            return response;
        }
        catch (error) {
            console.error('couchdb_service.postDocument(): Error creating document:', error);
        }
    }
    async putDocument(putDocumentParams) {
        try {
            const service = Nano(this.couchdbUrl);
            const db = service.use(putDocumentParams.db);
            // Modifiy the document into the database
            const resp = await db.insert(putDocumentParams.document);
            // resp will contain the _id and _rev of the newly created document
            const response = { 'result': resp };
            return response;
        }
        catch (error) {
            console.error('couchdb_service.putDocumentParams(): Error modifying document:', error);
        }
    }
    async getAttachment(getAttachmentParams) {
        try {
            const service = Nano(this.couchdbUrl);
            const db = service.use(getAttachmentParams.db);
            // get the attachment
            const resp = await db.attachment.get(getAttachmentParams.docId, getAttachmentParams.attachmentName);
            // 
            const response = { 'result': resp };
            return response;
        }
        catch (error) {
            console.error('couchdb_service.getAttachment(): Error getting attachment:', error);
        }
    }
    async deleteDocument(deleteDocumentParams) {
        try {
            const service = Nano(this.couchdbUrl);
            const db = service.use(deleteDocumentParams.db);
            // get the attachment
            const resp = await db.destroy(deleteDocumentParams.docId, deleteDocumentParams.rev);
            // 
            const response = { 'result': resp };
            return response;
        }
        catch (error) {
            console.error('couchdb_service.deleteDocument(): Error deleting document:', error);
        }
    }
    async postFind(postFindParams) {
        try {
            const service = Nano(this.couchdbUrl);
            const db = service.use(postFindParams.db);
            // find using selector
            const resp = await db.find(postFindParams.selector);
            // 
            const response = { 'result': resp };
            return response;
        }
        catch (error) {
            console.error('couchdb_service.postFind(): Error finding using selector:', error);
        }
    }
}
export const service = new CouchDBService('http://admin:c4ai-indl@104.154.155.83:5984');
//# sourceMappingURL=couchdb_service.js.map