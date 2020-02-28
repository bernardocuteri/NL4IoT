package it.unical.nl4iot.nl4iotframework.services;

import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;

import org.springframework.stereotype.Service;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import it.unical.nl4iot.nl4iotframework.model.NlAppValidationException;
import it.unical.nl4iot.nl4iotframework.model.NlApplication;
import it.unical.nl4iot.nl4iotframework.model.NlAwareHttpService;
import it.unical.nl4iot.nl4iotframework.model.intents.HttpIntentHandler;
import it.unical.nl4iot.nl4iotframework.model.intents.IntentHandler;
import it.unical.nl4iot.nl4iotframework.model.intents.IntentModel;
import it.unical.nl4iot.nl4iotframework.model.validator.ValidationResult;

@Service
public class JsonNLValidatorService {

	public final static String NL_AWARE_SERVICE_CONFIG_PATH = "getNLConfiguration";
	
	private NlAppJsonConsumer nlAppConsumer;

	public ValidationResult validate(String json) {
		try {
			System.out.println(json);
			JsonElement parsingElement = JsonParser.parseString(json);
			JsonObject appJsonObject = parsingElement.getAsJsonObject();
			System.out.println(appJsonObject);
			return validate(new Gson().fromJson(parsingElement, NlApplication.class));
		} catch(Exception e) {
			return new ValidationResult(ValidationResult.NOT_VALID, "Malformed JSON "+json);
		}
	}

	public boolean hasValue(String s) {
		return s != null && !s.isEmpty();
	}
	public void replaceHttpIntentHandlerDefaultService(List<IntentHandler> intentHandlers, String service) {
		for(IntentHandler handler : intentHandlers) {
			if(handler.getType().contentEquals(HttpIntentHandler.HTTP_INTENT_HANDLER_TYPE)) {
				HttpIntentHandler httpHanlder = (HttpIntentHandler) handler;
				if(!hasValue(httpHanlder.getBaseUrl())) {
					httpHanlder.setBaseUrl(service);
				}
			}
		}
	}

	public ValidationResult validate(NlApplication nlApplication) {
	
		if(nlApplication.getIntentHandlers() == null) {
			nlApplication.setIntentHandlers(new ArrayList<>());
		}
		if(nlApplication.getIntentModels() == null) {
			nlApplication.setIntentModels(new ArrayList<>());
		}
		if(nlApplication.getNlAwareHttpServices() == null) {
			nlApplication.setNlAwareHttpServices(new ArrayList<>());
		}
		
		try {
			List<IntentModel> intentModels = new LinkedList<>();
			intentModels.addAll(nlApplication.getIntentModels());
			
			//TODO replace with request parent path
			replaceHttpIntentHandlerDefaultService(nlApplication.getIntentHandlers(), "localhost");
			List<IntentHandler> intentHandlers = new LinkedList<>();
			intentHandlers.addAll(nlApplication.getIntentHandlers());
			
			validateNlCoreServiceUrl(nlApplication.getNlCoreService());
			
			Queue<String> nlAwareHttpServices = new LinkedList<>();
			Set<String> visitedNlAwareHttpService = new HashSet<>();
			for (NlAwareHttpService nlAwareService : nlApplication.getNlAwareHttpServices()) {
				validate(nlAwareService);
				nlAwareHttpServices.add(getNlAwareServiceURL(nlAwareService));
			}
			
			while(!nlAwareHttpServices.isEmpty()) {
				String nextNlAwareService = nlAwareHttpServices.poll();
				visitedNlAwareHttpService.add(nextNlAwareService);
				
				NlApplication serviceApp = nlAppConsumer.getNlApplication(nextNlAwareService, "get");
				//TODO proper validation
				intentModels.addAll(serviceApp.getIntentModels());
				
				replaceHttpIntentHandlerDefaultService(serviceApp.getIntentHandlers(), parentPath(nextNlAwareService));
				
				intentHandlers.addAll(serviceApp.getIntentHandlers());
				for(NlAwareHttpService nlAwareHttpServiceChild: serviceApp.getNlAwareHttpServices()) {
					validate(nlAwareHttpServiceChild);
					String childUrl = getNlAwareServiceURL(nlAwareHttpServiceChild);
					if(!visitedNlAwareHttpService.contains(childUrl)) {
						nlAwareHttpServices.add(childUrl);
					}
				}
				
				
				
			}
			NlApplication result = new NlApplication(intentModels, intentHandlers, new ArrayList<>(), nlApplication.getNlCoreService());
			return new ValidationResult(ValidationResult.VALID, result);
		} catch (NlAppValidationException e) {
			return new ValidationResult(ValidationResult.NOT_VALID, e.getMessage());
		}
	}

	private String parentPath(String url) {
		return url.substring(0, url.lastIndexOf("/"));
	}

	private void validateNlCoreServiceUrl(String nlCoreService) throws NlAppValidationException {
		validateUrl(nlCoreService);
	}

	public String getNlAwareServiceURL(NlAwareHttpService nlAwareService) {
		if (hasValue(nlAwareService.getUrl())) {
			return nlAwareService.getUrl();
		}
		return getUrlFromBase(nlAwareService.getBaseUrl());
	}

	private void validate(NlAwareHttpService nlAwareService) throws NlAppValidationException {
		
		if (hasValue(nlAwareService.getUrl()) && hasValue(nlAwareService.getBaseUrl())
				&& !compatibleURLs(nlAwareService.getUrl(), nlAwareService.getBaseUrl()))
			throw new NlAppValidationException("use either baseUrl or url for nlAwareHtppServices");
		
		String nlAwareServiceURL = getNlAwareServiceURL(nlAwareService);
		String httpMethod = nlAwareService.getHttpMethod();
		if (!httpMethod.isEmpty() && !nlAwareService.getHttpMethod().equalsIgnoreCase("get") && !nlAwareService.getHttpMethod().equalsIgnoreCase("post"))
			throw new NlAppValidationException("invalid http method " + httpMethod + " for nlAwareService " + nlAwareServiceURL);
		validateUrl(nlAwareServiceURL);

	}

	private void validateUrl(String urlCandidate) throws NlAppValidationException {
		try {
			URL u = new URL(urlCandidate);
			u.toURI();
		} catch (URISyntaxException | MalformedURLException e) {
			throw new NlAppValidationException("Illegarl URL " + urlCandidate);
		}
	}

	public String getUrlFromBase(String baseUrl) {
		if (baseUrl.endsWith("/")) {
			baseUrl = baseUrl.substring(0, baseUrl.length() - 1);
		}
		return baseUrl + "/" + NL_AWARE_SERVICE_CONFIG_PATH;
	}

	private boolean compatibleURLs(String url, String baseUrl) {
		return url.equals(getUrlFromBase(baseUrl));
	}

}
