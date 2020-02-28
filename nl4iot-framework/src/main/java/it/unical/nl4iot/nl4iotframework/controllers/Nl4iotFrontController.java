package it.unical.nl4iot.nl4iotframework.controllers;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import it.unical.nl4iot.nl4iotframework.model.NlApplication;
import it.unical.nl4iot.nl4iotframework.model.intents.Intent;
import it.unical.nl4iot.nl4iotframework.model.intents.IntentResult;
import it.unical.nl4iot.nl4iotframework.model.validator.ValidationResult;
import it.unical.nl4iot.nl4iotframework.services.JsonNLValidatorService;
import it.unical.nl4iot.nl4iotframework.services.NLService;


@Controller
public class Nl4iotFrontController {
	
	@Autowired
	private JsonNLValidatorService jsonValidatorService;
	
	@Autowired
	private NLService nlService;
	
	public Object trainApplication(NlApplication app) {
		//app.nl
		return null;	
	}
	
	public Intent getIntent(String nlPrhase, NlApplication app) {
		return nlService.getIntent(nlPrhase, app);
	}
	
	public Intent getIntent(String nlPrhase, String nlInferenceModel) {
		return null;
	}
	
	public IntentResult handleNLPhrase(String nlPhrase, NlApplication app) {
		return null;
	}
	
	@PostMapping(value ="/validate-nl-app-json", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
	@ResponseBody
	public ValidationResult validateNLAppJson(HttpEntity<String> httpEntity) {
		System.out.println(httpEntity.getHeaders());
		return jsonValidatorService.validate(httpEntity.getBody());
	}
	
//	@PostMapping(value ="/validate-nl-app-json", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
//	@ResponseBody
//	public ValidationResult validateNLAppJson(@RequestBody NlApplication nlApplication) {
//		System.out.println(nlApplication.getIntentModels().size());
//		return jsonValidatorService.validate(nlApplication);
//	}

}