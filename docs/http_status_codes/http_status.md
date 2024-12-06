# HTTP Status Codes Reference

When using Lorax, you may sometimes receive confusing HTTP messages.  
Refer to this file for definitions of HTTP status codes.

---

## 1xx: Informational Responses

- **100 Continue**: The initial part of the request was received, and the client can continue.
- **101 Switching Protocols**: The server is switching protocols as requested.
- **102 Processing**: The server is processing the request but has not completed it yet (WebDAV).
- **103 Early Hints**: Used to return some response headers before the final HTTP message.

---

## 2xx: Success

- **200 OK**: The request was successful.
- **201 Created**: The request succeeded, and a resource was created.
- **202 Accepted**: The request has been accepted for processing, but not completed.
- **203 Non-Authoritative Information**: The returned meta-information is not from the origin server.
- **204 No Content**: The server successfully processed the request but returned no content.
- **205 Reset Content**: The client should reset the document view.
- **206 Partial Content**: The server is delivering only part of the resource due to a range header sent by the client.
- **207 Multi-Status**: Conveys multiple status codes for a single request (WebDAV).
- **208 Already Reported**: The members of a DAV binding have already been enumerated (WebDAV).
- **226 IM Used**: The server has fulfilled a GET request for the resource (HTTP Delta Encoding).

---

## 3xx: Redirection

- **300 Multiple Choices**: Multiple options for the resource are available.
- **301 Moved Permanently**: The URL of the requested resource has been permanently changed.
- **302 Found**: The resource resides temporarily under a different URL.
- **303 See Other**: The response to the request can be found under another URL.
- **304 Not Modified**: Indicates that the cached version is still valid.
- **305 Use Proxy**: The requested resource must be accessed through a proxy (deprecated).
- **306 (Unused)**: Previously used, but no longer in use.
- **307 Temporary Redirect**: The resource temporarily resides under a different URL, and the method used should not be changed.
- **308 Permanent Redirect**: The resource is now permanently located at another URL.

---

## 4xx: Client Errors

- **400 Bad Request**: The request cannot be fulfilled due to bad syntax.
- **401 Unauthorized**: Authentication is required and has failed or has not been provided.
- **402 Payment Required**: Reserved for future use.
- **403 Forbidden**: The request was valid, but the server is refusing action.
- **404 Not Found**: The requested resource could not be found.
- **405 Method Not Allowed**: The request method is not supported for the requested resource.
- **406 Not Acceptable**: The requested resource is capable of generating only unacceptable content.
- **407 Proxy Authentication Required**: The client must authenticate itself with the proxy.
- **408 Request Timeout**: The server timed out waiting for the request.
- **409 Conflict**: Indicates a request conflict with the current state of the server.
- **410 Gone**: The resource is no longer available and will not be available again.
- **411 Length Required**: The server refuses to accept the request without a valid `Content-Length` header.
- **412 Precondition Failed**: The server does not meet one of the preconditions set by the client.
- **413 Payload Too Large**: The request entity is larger than the server is willing or able to process.
- **414 URI Too Long**: The URI provided was too long for the server to process.
- **415 Unsupported Media Type**: The media format of the requested data is not supported by the server.
- **416 Range Not Satisfiable**: The range specified by the `Range` header field cannot be fulfilled.
- **417 Expectation Failed**: The server cannot meet the requirements of the `Expect` header field.
- **418 I'm a Teapot**: A playful response defined in RFC 2324 (April Fools').
- **421 Misdirected Request**: The request was directed to a server that is unable to produce a response.
- **422 Unprocessable Entity**: The request was well-formed but unable to be followed due to semantic errors (WebDAV).
- **423 Locked**: The resource being accessed is locked (WebDAV).
- **424 Failed Dependency**: The request failed due to failure of a previous request (WebDAV).
- **425 Too Early**: The server is unwilling to risk processing a request that might be replayed.
- **426 Upgrade Required**: The client should switch to a different protocol.
- **428 Precondition Required**: The origin server requires the request to be conditional.
- **429 Too Many Requests**: The user has sent too many requests in a given amount of time.
- **431 Request Header Fields Too Large**: The server refuses to process the request due to large headers.
- **451 Unavailable For Legal Reasons**: The resource is unavailable due to legal reasons.

---

## 5xx: Server Errors

- **500 Internal Server Error**: A generic error occurred on the server.
- **501 Not Implemented**: The server does not recognize the request method.
- **502 Bad Gateway**: The server received an invalid response from the upstream server.
- **503 Service Unavailable**: The server is not ready to handle the request.
- **504 Gateway Timeout**: The upstream server failed to send a request in time.
- **505 HTTP Version Not Supported**: The server does not support the HTTP protocol version.
- **506 Variant Also Negotiates**: The server has an internal configuration error.
- **507 Insufficient Storage**: The server is unable to store the representation (WebDAV).
- **508 Loop Detected**: The server detected an infinite loop while processing the request (WebDAV).
- **510 Not Extended**: Further extensions to the request are required.
- **511 Network Authentication Required**: The client needs to authenticate to gain network access.