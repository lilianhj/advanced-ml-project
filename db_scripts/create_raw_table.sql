CREATE TABLE raw_data (dab_id VARCHAR(7) PRIMARY KEY,
                       alj_id VARCHAR(7),
                       dab_text TEXT,
                       alj_text TEXT,
                       dab_url VARCHAR(255),
                       alj_url VARCHAR(255),
                       decision_text TEXT,
                       decision_binary INTEGER,
                       dab_year INTEGER,
                       alj_year INTEGER);
CREATE INDEX ON raw_data (dab_id);
CREATE INDEX ON raw_data (alj_id);
