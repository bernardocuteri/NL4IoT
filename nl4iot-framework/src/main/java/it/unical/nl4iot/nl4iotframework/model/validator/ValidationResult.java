package it.unical.nl4iot.nl4iotframework.model.validator;

import it.unical.nl4iot.nl4iotframework.model.NlApplication;

public class ValidationResult {

	public static final String VALID = "VALID";

	public static final String NOT_VALID = "NOT_VALID";

	private String validation;

	private String error;

	private NlApplication nlApp;

	public ValidationResult() {
	}

	public ValidationResult(String validation, NlApplication nlApp) {
		super();
		this.validation = validation;
		this.nlApp = nlApp;
	}

	public ValidationResult(String validation, String error) {
		super();
		this.validation = validation;
		this.error = error;
	}

	public String getValidation() {
		return validation;
	}

	public void setValidation(String validation) {
		this.validation = validation;
	}

	public String getError() {
		return error;
	}

	public void setError(String error) {
		this.error = error;
	}

	public NlApplication getNlApp() {
		return nlApp;
	}

	public void setNlApp(NlApplication nlApp) {
		this.nlApp = nlApp;
	}

}
