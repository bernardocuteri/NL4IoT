package it.unical.nl4iot.nl4iotframework.model;

import java.util.List;

import it.unical.nl4iot.nl4iotframework.model.intents.IntentHandler;
import it.unical.nl4iot.nl4iotframework.model.intents.IntentModel;

/**
 * @author bernardo
 *
 */
public class NlApplication {

	private List<IntentModel> intentModels;

	private List<IntentHandler> intentHandlers;

	private List<NlAwareHttpService> nlAwareHttpServices;

	private String nlCoreService;

	public NlApplication() {
		super();
	}

	public NlApplication(List<IntentModel> intentModels, List<IntentHandler> intentHandlers,
			List<NlAwareHttpService> nlAwareHttpServices, String nlCoreService) {
		super();
		this.intentModels = intentModels;
		this.intentHandlers = intentHandlers;
		this.nlAwareHttpServices = nlAwareHttpServices;
		this.nlCoreService = nlCoreService;
	}

	public List<IntentModel> getIntentModels() {
		return intentModels;
	}

	public void setIntentModels(List<IntentModel> intentModels) {
		this.intentModels = intentModels;
	}

	public List<IntentHandler> getIntentHandlers() {
		return intentHandlers;
	}

	public void setIntentHandlers(List<IntentHandler> intentHandlers) {
		this.intentHandlers = intentHandlers;
	}

	public List<NlAwareHttpService> getNlAwareHttpServices() {
		return nlAwareHttpServices;
	}

	public void setNlAwareHttpServices(List<NlAwareHttpService> nlAwareHttpServices) {
		this.nlAwareHttpServices = nlAwareHttpServices;
	}

	public String getNlCoreService() {
		return nlCoreService;
	}

	public void setNlCoreService(String nlCoreService) {
		this.nlCoreService = nlCoreService;
	}

}
