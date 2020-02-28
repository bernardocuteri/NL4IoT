package it.unical.nl4iot.nl4iotframework.model.intents;

public class IntentHandler {

	private String type;

	private String intent;

	public IntentHandler() {
		super();
	}

	public IntentHandler(String type, String intent) {
		super();
		this.type = type;
		this.intent = intent;
	}

	public String getIntent() {
		return intent;
	}

	public void setIntent(String intent) {
		this.intent = intent;
	}

	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}

}
