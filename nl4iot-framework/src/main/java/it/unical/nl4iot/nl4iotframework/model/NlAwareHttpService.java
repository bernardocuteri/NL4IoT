package it.unical.nl4iot.nl4iotframework.model;

public class NlAwareHttpService {

	// url is baseUrl + "getNLConfiguration
	private String url;

	private String baseUrl;
	
	//GET or POST 
	private String httpMethod;

	public NlAwareHttpService() {
		super();
	}

	public NlAwareHttpService(String url, String baseUrl) {
		super();
		this.url = url;
		this.baseUrl = baseUrl;
		this.httpMethod = "GET";
	}

	public String getUrl() {
		return url;
	}

	public void setUrl(String url) {
		this.url = url;
	}

	public String getBaseUrl() {
		return baseUrl;
	}

	public void setBaseUrl(String baseUrl) {
		this.baseUrl = baseUrl;
	}

	public String getHttpMethod() {
		return httpMethod;
	}

	public void setHttpMethod(String httpMethod) {
		this.httpMethod = httpMethod;
	}
	
}
