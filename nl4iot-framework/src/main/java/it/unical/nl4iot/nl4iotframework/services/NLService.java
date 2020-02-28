package it.unical.nl4iot.nl4iotframework.services;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import it.unical.nl4iot.nl4iotframework.model.NlApplication;
import it.unical.nl4iot.nl4iotframework.model.intents.Intent;
import it.unical.nl4iot.nl4iotframework.model.validator.ValidationResult;

@Service
public class NLService {

	@Autowired
	private JsonNLValidatorService jsonNLValidatorService;

	public Intent getIntent(String nlPrhase, NlApplication app) {
		ValidationResult validation = jsonNLValidatorService.validate(app);
		if (validation.getValidation().equals(ValidationResult.NOT_VALID)) {
			return null;
		}
		return null;

	}

}
