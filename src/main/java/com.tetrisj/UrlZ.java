package com.tetrisj;

import org.apache.commons.lang.NullArgumentException;
import org.apache.commons.lang.StringEscapeUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.builder.EqualsBuilder;
import org.apache.commons.lang.builder.HashCodeBuilder;
import com.google.common.net.InternetDomainName;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.*;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * UrlZ should provide all url services for SimilarGroup java projects.
 *
 * @author yaniv hadad
 */
public class UrlZ{

    private static final char TOTAL_FLD_SUFFIX_C = '*';
    private static final char SECOND_LEVEL_DOMAIN_SUFFIX_C = '#';

    private static final String PROTOCOL_PREFIX = "://";
    private static final String DEFAULT_PREFIX = "http://";
    private static final String HTTPS_PREFIX = "https://";
    private static final String TOTAL_FLD_SUFFIX = Character.toString(TOTAL_FLD_SUFFIX_C);
    private static final String SECOND_LEVEL_DOMAIN_SUFFIX = Character.toString(SECOND_LEVEL_DOMAIN_SUFFIX_C);
    private static final String WWW = "www";
    private static final String WWW_PREFFIX = WWW + ".";

    private static final String HTTP_PROTOCOL = "http";
    private static final String HTTPS_PROTOCOL = "https";

    private static final String TRIM_URL_SUFFIX_CHARS = "/#?&";
    private static Pattern _splitPat = Pattern.compile("[&#]");
    private static Pattern _slash = Pattern.compile("/");

    private String _originalUrl;
    private String _rawUrl;
    private String _page;
    private String[] _queryParams;
    private String _rawPage;
    private String _shortUrl;
    private String _topLevelDomain;
    private boolean _isCountryDomain;
    private String _mainDomain;
    private String _domain;
    private String _secondLevelDomain;
    private String _secondLevelDomainKey;
    private String _firstLevelDomain;
    private String _firstLevelDomainKey;
    private String _subDomain;
    private String _uniqueUrl;
    private String _protocol;
    private String _publicSuffix;
    private URL _siteUrl;
    private InternetDomainName _domainName;
    private List<String> _folders;
    private boolean _isFirstLevelDomain;
    private boolean _isSecondLevelDomain;

    /**
     * Return the folders of the url. (Lazy load)<br>
     * Example 1: www.google.com/a/b/c will return { "a", "b", "c" }<br>
     * Example 2: www.google.com will return Empty list.
     *

     */
    private List<String> getFolders() {

        if (_folders == null) fillFolders();

        return _folders;
    }

    /**
     * Will return the protocol of the url.<br>
     * Currently it supports only http or https.<br>
     * Example 1: For google.com will return http<br>
     * Example 2: For http://google.com will return http<br>
     * Example 3: For https://google.com will return https
     *

     */
    private String getProtocol() {
        return _protocol;
    }

    private String getPage() {
        return _page;
    }

    /**
     * Return the page of the url.<br>
     * Example 1: For similarweb.com/yaniv?b=a&a=b#c=d	will return /yaniv?b=a&a=b#c=d
     *

     */
    private String getRawPage() {
        return _rawPage;
    }

    /**
     * Return the raw url that UrlZ was built upon after cleaning and fixing it as follows.<br>
     * 1. Double decoding the url.<br>
     * 2. Remove illegal chars (\t,\n,\r,\0)<br>
     * 3. Add http:// to the beginning if no protocol specified.<br>
     *

     */
    public String getRawUrl() {
        return _rawUrl;
    }

    public String[] getQueryParams() {

        return _queryParams;
    }

    /**
     * Return Subdomain.FirstLevelDomain<br>
     * Example 1: For http://www.google.com will return google.com<br>
     * Example 2: For a.b.c.d.com will return a.b.c.d.com<br>
     * Example 3: For www.a.b.c.d.com will return www.a.b.c.d.com<br>
     *

     */
    public String getShortUrl() {
        return _shortUrl;
    }

    /**
     * Return Subdomain.FirstLevelDomain<br>
     * If there is no subdomain than return www.FirstLevelDomain.
     * Example 1: For http://www.google.com will return http://www.google.com<br>
     * Example 2: For a.b.c.d.com will return http://a.b.c.d.com<br>
     * Example 3: For www.a.b.c.d.com will return http://www.a.b.c.d.com<br>
     * Example 3: For https://www.a.b.c.d.com will return https://www.a.b.c.d.com<br>
     *

     */
    public String getShortUrlWithWWWAndHTTP() {

        return _protocol + PROTOCOL_PREFIX + getShortUrlWithWWW();
    }

    /**
     * Return Subdomain.FirstLevelDomain<br>
     * If there is no subdomain than return www.FirstLevelDomain.
     * Example 1: For http://www.google.com will return www.google.com<br>
     * Example 2: For a.b.c.d.com will return a.b.c.d.com<br>
     * Example 3: For www.a.b.c.d.com will return www.a.b.c.d.com<br>
     *

     */
    private String getShortUrlWithWWW() {
        if (getIsFirstLevelDomain()) {
            return UrlZ.WWW_PREFFIX + _shortUrl;
        } else {
            return _shortUrl;
        }
    }

    /**
     * Will return the Top Level Domain (TLD) for the url.<br>
     * Example 1: For www.google.com will return com.<br>
     * Example 2: For www.google.co.il will return il.<br>
     * Example 3: For snap.do will return do.<br>
     *

     */
    public String getTopLevelDomain() {
        return _topLevelDomain;
    }

    /**
     * Will return the Public Suffix for the url.<br>
     * Example 1: For www.google.com will return com.<br>
     * Example 2: For www.google.co.il will return co.il.<br>
     * Example 3: For snap.do will return do.<br>
     *

     */
    public String getPublicSuffix() {
        return _publicSuffix;
    }

    /**
     * Will return True if the top level domain is country TLD else otherwise.<br>
     * Note: The current logic says that if the Top Level Domain is 2 chars in length then its a country TLD.<br>
     * Example 1: For www.google.com Will return False.<br>
     * Example 2: For www.google.co.il Will return True.<br>
     * Example 3: For www.google.edu.uk Will return True.<br>
     * Example 4: For snap.do Will return True.<br>
     * Example 5: For www.google.net Will return False.<br>
     *

     */
    public boolean getIsCountryDomain() {
        return _isCountryDomain;
    }

    /**
     * Will return the main domain for the url.<br>
     * Example 1: For www.google.com will return google<br>
     * Example 2: For google.com will return google<br>
     * Example 3: For a.b.c.d.com/b/c/d will return d<br>
     *
 Main domain
     */
    public String getMainDomain() {
        return _mainDomain;
    }

    /**
     * Will return the domain for the url.<br>
     * Example 1: For www.google.com will return google<br>
     * Example 2: For google.com will return google<br>
     * Example 3: For a.b.c.d.com/b/c/d will return a.b.c.d<br>
     *
 Main domain
     */
    public String getDomain() {
        return _domain;
    }

    /**
     * Will return the Second Level Domain (SLD) of the url.<br>
     * Example 1: For www.google.com will return null<br>
     * Example 2: For http://mail.google.com will return mail.google.com<br>
     * Example 3: For http://test.mail.google.com will return mail.google.com<br>
     *

     */
    public String getSecondLevelDomain() {
        return _secondLevelDomain;
    }

    /**
     * Will return the Second Level Domain key of the url.<br>
     * Example 1: For www.google.com will return null<br>
     * Example 2: For http://mail.google.com will return mail.google.com#<br>
     * Example 3: For http://test.mail.google.com will return mail.google.com#<br>
     *

     */
    public String getSecondLevelDomainKey() {
        return _secondLevelDomainKey;
    }

    /**
     * Will return the First Level Domain (FLD) of the url.<br>
     * Example 1: For www.google.com will return google.com<br>
     * Example 2: For http://mail.google.com will return google.com<br>
     *

     */
    public String getFirstLevelDomain() {
        return _firstLevelDomain;
    }

    /**
     * Will return the First Level Domain of the url.<br>
     * Example 1: For www.google.com will return google.com<br>
     * Example 2: For http://mail.google.com will return google.com*<br>
     *

     */
    public String getFirstLevelDomainKey() {
        return _firstLevelDomainKey;
    }

    /**
     * Will return the sub domain for the underline url and null if there is no subdomain<br>
     * Example 1: For http://www.google.com will return null<br>
     * Example 2: For a.b.com will return a<br>
     * Example 3: For a.b.c.d.com will return a.b.c<br>
     * Example 4: For www.a.b.com will return www.a.b<br>
     *

     */
    private String getSubDomain() {
        return _subDomain;
    }

    /**
     * Return True if the url domain is first level domain, False otherwise.<br>
     * Example 1: For www.google.com return True.<br>
     * Example 2: For a.google.com return False.<br>
     * Example 3: For a.b.google.com return False.<br>
     * Example 4: For http://www.google.com/blabla?u=3 return True.<br>
     * Example 5: For http://search.google.com return False.<br>
     *

     */
    private boolean getIsFirstLevelDomain() {
        return _isFirstLevelDomain;
    }

    /**
     * Return True if the url domain is first level domain, False otherwise.<br>
     * Example 1: For www.google.com return False.<br>
     * Example 2: For a.google.com return True.<br>
     * Example 3: For a.b.google.com return False.<br>
     * Example 4: For http://www.google.com/blabla?u=3 return False.<br>
     * Example 5: For http://search.google.com return True.<br>
     *

     */
    private boolean getIsSecondLevelDomain() {
        return _isSecondLevelDomain;
    }

    /**
     * Will return true if the url is a subdomain, false otherwise.

     */
    public boolean isSubdomain() {
        return StringUtils.isNotBlank(getSubDomain());
    }

    /**
     * Will return True if the url is a homepage and False otherwise.
     * Example 1: For www.google.com will return True.<br>
     * Example 2: For mail.google.com will return True.<br>
     * Example 3: For http://google.com will return True.<br>
     * Example 4: For www.google.com?a=b will return False.<br>
     * Example 5: For www.google.com/a/b will return False.<br>
     *

     */
    public boolean getIsHomepage() {
        return (_siteUrl.getQuery() == null && (_siteUrl.getPath().equals("") || _siteUrl.getPath().equals("/")));
    }

    /**
     * Will return the original url that UrlZ was created with<br>
     * without any modifications.
     *

     */
    private String getOriginalUrl() {
        return _originalUrl;
    }

    public URL getURL() {
        return _siteUrl;
    }

    public String getPath() {
        return _siteUrl.getPath();
    }

    /**
     * Will return the host of the underline url.<br>
     * Example 1: For http://www.google.com will return www.google.com<br>
     * Example 2: For http://www.a.b.c.com will return www.a.b.c.com<br>
     * Example 3: For google.com will return google.com<br>
     * Example 4: For http://www.google.com/one/two?three=3 will return www.google.com<br>
     *

     */
    public String getHost() {
        return _siteUrl.getHost();
    }

    public UrlZ() {
    }

    public UrlZ(String rawUrl) throws MalformedURLException {
        initUrlEntity(rawUrl);
    }

    public UrlZ(URL url) throws MalformedURLException {
        _originalUrl = url.toString();
        _rawUrl = url.toString();
        _siteUrl = url;

        if (_siteUrl.getHost().toLowerCase().startsWith("www.")) {
            _siteUrl = new URL(_siteUrl.getProtocol(), StringUtils.removeStart(_siteUrl.getHost().toLowerCase(), "www."), _siteUrl.getFile());
        }

        _domainName = validateAndReturnInternetDomainName(_siteUrl.getHost().toLowerCase());

        fillData();
    }

    private void initUrlEntity(String url) throws MalformedURLException {
        _originalUrl = url;

        innerValidate(url);

        fillData();
    }

    private void innerValidate(String url) throws MalformedURLException {
        if (url.length() > 4096) {
            throw new MalformedURLException("URL too long: " + url.length());
        }

        StringBuffer protocol = new StringBuffer();

        _rawUrl = validateAndFixUrl(url, protocol);
        _protocol = protocol.toString();

        _siteUrl = validateAndReturnBasicUrlJavaObject(_rawUrl, _protocol);

        _domainName = validateAndReturnInternetDomainName(_siteUrl.getHost().toLowerCase());
    }

    public static String removeKeySuffixFromUrl(String url) {

        // If aggregated first level domain site we remove the * and create a put.
        if (url.endsWith(UrlZ.TOTAL_FLD_SUFFIX)) {
            url = url.substring(0, url.length() - 1);
        }
        // If aggregated second level domain site we remove the # and create a put.
        else if (url.endsWith(UrlZ.SECOND_LEVEL_DOMAIN_SUFFIX)) {
            url = url.substring(0, url.length() - 1);
        }

        return url;
    }

    /**
     * Check if the url is a valid based on UrlZ validations.
     *
     * @param rawUrl
 true if the url is valid, False otherwise
     */
    public static boolean validate(String rawUrl) {
        try {
            rawUrl = validateAndFixUrl(rawUrl, null);
            URL siteUrl = validateAndReturnBasicUrlJavaObject(rawUrl);
            validateAndReturnInternetDomainName(siteUrl.getHost().toLowerCase());

            return true;
        } catch (Exception e) {
        }

        return false;
    }

    /**
     * Will try to create urlz instance but will not throw exception if it fails.
     *
     * @param url
 urlz instance if the url is valid, null otherwise.
     */
    public static UrlZ tryCreate(String url) {
        UrlZ urlZ = null;

        try {
            urlZ = new UrlZ(url);
        } catch (Exception e) {
        }

        return urlZ;
    }

    private static String cleanUrlSuffix(String url) {

        return StringUtils.stripEnd(url, TRIM_URL_SUFFIX_CHARS);
    }

    private static String cleanUrlPrefix(String url, StringBuffer protocol) {

        if (url.startsWith(DEFAULT_PREFIX)) {
            url = url.substring(DEFAULT_PREFIX.length());
            if (protocol != null) protocol.append(HTTP_PROTOCOL);
        } else if (url.startsWith(HTTPS_PREFIX)) {
            url = url.substring(HTTPS_PREFIX.length());
            if (protocol != null) protocol.append(HTTPS_PROTOCOL);
        } else {
            if (protocol != null) protocol.append(HTTP_PROTOCOL);
        }

        while (url.startsWith(WWW_PREFFIX)) {
            url = url.substring(WWW_PREFFIX.length());
        }

        return url;
    }

    private static String doubleHtmlUnescape(String url) {
        int firstAmp = url.indexOf(';');
        if (firstAmp > 0) {
            url = StringEscapeUtils.unescapeHtml(url);

            firstAmp = url.indexOf(';');

            if (firstAmp > 0) {
                url = StringEscapeUtils.unescapeHtml(url);
            }
        }

        return url;
    }

    private static String getUrlWithDomainInLowerCase(String url) {

        int indexOfProtocol = url.indexOf("//");
        int startIndex = 0;

        // Make sure the // is really belongs to the protocol and not in the page part
        if (indexOfProtocol > 0 && indexOfProtocol < 8) {
            startIndex = indexOfProtocol + 2;
        }

        int indexOfDomainEnd = url.indexOf("/", startIndex);

        if (indexOfDomainEnd < 0) {
            return url.toLowerCase();
        } else {

            String urlWitDomainInLowerCase = StringUtils.substring(url, 0, indexOfDomainEnd).toLowerCase() + StringUtils.substring(url, indexOfDomainEnd);

            return urlWitDomainInLowerCase;
        }
    }

    private static String validateAndFixUrl(String url, StringBuffer protocol) throws MalformedURLException {

        if (StringUtils.isBlank(url)) throw new NullArgumentException("rawUrl");

        url = getUrlWithDomainInLowerCase(url);

        // Remove http://, https:// and www. from the start of the url
        url = cleanUrlPrefix(url, protocol);

        // Remove ?, / and # at the end
        url = cleanUrlSuffix(url);

        // we try to double unescape the URL to overcome problems like "&amp;amp;" that needs to be "&"
        url = doubleHtmlUnescape(url);

        // Clean tabs, enters and EOF chars from the url
        url = cleanUrlIllegalChars(url);

        return url;
    }

    private static URL validateAndReturnBasicUrlJavaObject(String url) throws MalformedURLException {
        return validateAndReturnBasicUrlJavaObject(url, HTTP_PROTOCOL);
    }

    private static URL validateAndReturnBasicUrlJavaObject(String url, String protocol) throws MalformedURLException {
        URL siteUrl = new URL(protocol + PROTOCOL_PREFIX + url);

        if (siteUrl.getHost().endsWith("."))
            throw new MalformedURLException("Host ends with '.' - " + url);

        if (StringUtils.isNotBlank(siteUrl.getUserInfo()))
            throw new MalformedURLException("Host contains UserInfo ('@') - " + url);

        return siteUrl;
    }

    private static InternetDomainName validateAndReturnInternetDomainName(String url) throws MalformedURLException {
        InternetDomainName domainName;

        try {
            domainName = InternetDomainName.from(url);
        } catch (Exception e) {
            throw new MalformedURLException(e.getMessage());
        }

        //if (!domainName.hasPublicSuffix() || domainName.getPublicSuffixIndex() == 0) {
        if (!domainName.hasPublicSuffix()) {
            throw new MalformedURLException("have no public suffix: " + url);
        }

        if (domainName.toString().indexOf('.') == -1) {
            throw new MalformedURLException("no dot: " + url);
        }

        return domainName;
    }

    private void fillData() throws MalformedURLException {

        String url = _siteUrl.getHost().toLowerCase();

        InternetDomainName topPrivateDomain;

        if (!_domainName.isUnderPublicSuffix() && _domainName.hasPublicSuffix()) {
            topPrivateDomain = _domainName;
        } else {
            topPrivateDomain = _domainName.topPrivateDomain();
        }

        // First Level Domain
        _firstLevelDomain = topPrivateDomain.name();
        _firstLevelDomainKey = _firstLevelDomain + TOTAL_FLD_SUFFIX;

        // Main Domain
        _mainDomain = topPrivateDomain.parts().get(0);

        String domainName = _domainName.name();

        if (_domainName.isUnderPublicSuffix()) {
            // Maybe the (-1) is not good for blogspot
            _domain = domainName.substring(0, domainName.length() - _domainName.publicSuffix().name().length() - 1);
            _publicSuffix = _domainName.publicSuffix().toString();
        }

        // Sub Domain
        if (_domainName.parts().size() <= topPrivateDomain.parts().size()) {
            _subDomain = null;

        } else {
            _subDomain = domainName.substring(0, domainName.length() - topPrivateDomain.name().length() - 1);
        }

        // Short Url & Is First Level Domain & Is Second Level Domain
        if (_subDomain == null) {
            _shortUrl = _firstLevelDomain;
            _isFirstLevelDomain = true;
        } else {
            _shortUrl = url;

            InternetDomainName secondPrivateDomain = _domainName;

            while (secondPrivateDomain.parts().size() - 1 > topPrivateDomain.parts().size())
                secondPrivateDomain = secondPrivateDomain.parent();

            _isSecondLevelDomain = (secondPrivateDomain.parts().size() == _domainName.parts().size());

            _secondLevelDomain = secondPrivateDomain.name();
            _secondLevelDomainKey = _secondLevelDomain + SECOND_LEVEL_DOMAIN_SUFFIX;
        }

        _topLevelDomain = _shortUrl.substring(_shortUrl.lastIndexOf(".") + 1);

        _isCountryDomain = (_topLevelDomain.length() == 2);

        if (_rawUrl.contains("?")) {
            String queryParamsString = _rawUrl.substring(_rawUrl.indexOf("?"));

            if (queryParamsString.length() > 1) {
                _queryParams = _splitPat.split(queryParamsString.substring(1));
            }
        }

        _page = _siteUrl.getFile();

        if (_rawUrl.contains("#")) {
            _rawPage = _page + _rawUrl.substring(_rawUrl.indexOf("#"));
        } else {
            _rawPage = _page;
        }
    }

    private void fillFolders() {

        _folders = new ArrayList<String>();

        int startIndex = 0;

        int endIndex = StringUtils.indexOfAny(_rawUrl, "?#");

        if (endIndex == -1) endIndex = _rawUrl.length();

        String urlFreg[] = _slash.split(_rawUrl.substring(startIndex, endIndex));

        //Start running from the first dir to query params (?) or (#)
        for (int i = 1; i < urlFreg.length; i++)
            _folders.add(urlFreg[i]);
    }

//	private static String cleanUrlIllegalChars(String url) {
//		return url.replace("\t","").replace("\n","").replace("\r","").replace("\0", "");
//	}

    private static Pattern removePattern = Pattern.compile("[\t\n\r\0]");

    private static String cleanUrlIllegalChars(String url) {
        return removePattern.matcher(url).replaceAll("");
    }


    @Override
    public String toString() {
        return getRawUrl();
    }

    @Override
    public boolean equals(Object obj) {

        if (obj == null)
            return false;
        if (obj == this)
            return true;
        if (obj.getClass() != getClass())
            return false;

        UrlZ rhs = (UrlZ) obj;

        return new EqualsBuilder().
                append(_rawUrl, rhs._rawUrl).
                isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder(31, 43). // two randomly chosen prime numbers
                append(_rawUrl).toHashCode();
    }

    public void clear() {
        _rawUrl = StringUtils.EMPTY;
        _page = StringUtils.EMPTY;
        _rawPage = StringUtils.EMPTY;
        _shortUrl = StringUtils.EMPTY;
        _topLevelDomain = StringUtils.EMPTY;
        _mainDomain = StringUtils.EMPTY;
        _secondLevelDomain = StringUtils.EMPTY;
        _secondLevelDomainKey = StringUtils.EMPTY;
        _firstLevelDomain = StringUtils.EMPTY;
        _firstLevelDomainKey = StringUtils.EMPTY;
        _subDomain = StringUtils.EMPTY;
        _uniqueUrl = StringUtils.EMPTY;
        _protocol = StringUtils.EMPTY;
        _queryParams = null;
        _siteUrl = null;
        _domainName = null;
        _folders = null;
        _isCountryDomain = false;
        _isFirstLevelDomain = false;
        _isSecondLevelDomain = false;
    }

    public static String encodeUri(String uri) {
        try {
            return URLEncoder.encode(uri, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

    public static String decodeUri(String uri) {
        try {
            return URLDecoder.decode(uri, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

    public static UrlZ combineUrl(UrlZ base, String path) throws MalformedURLException {
        URL url = new URL(base._siteUrl, path);
        return new UrlZ(url);
    }

    public static void main(String[] args) throws MalformedURLException, URISyntaxException {

        String url = args[0];

        UrlZ urlz = new UrlZ(url);

        System.out.println("Original Url:\t" + urlz.getOriginalUrl());
        System.out.println("Url with http:\t" + urlz.getShortUrlWithWWWAndHTTP());
        System.out.println("Raw Url:\t" + urlz.getRawUrl());
        System.out.println("Main Domain:\t" + urlz.getMainDomain());
        System.out.println("Top Level Domain:\t" + urlz.getTopLevelDomain());
        System.out.println("Domain:\t" + urlz.getDomain());
        System.out.println("Short Url:\t" + urlz.getShortUrl());
        System.out.println("Sub Domain:\t" + urlz.getSubDomain());
        System.out.println("Protocol:\t" + urlz.getProtocol());
        System.out.println("Is First Level Domain:\t" + urlz.getIsFirstLevelDomain());
        System.out.println("Is Second Level Domain:\t" + urlz.getIsSecondLevelDomain());
        System.out.println("First Level Domain:\t" + urlz.getFirstLevelDomain());
        System.out.println("Second Level Domain:\t" + urlz.getSecondLevelDomain());
        System.out.println("First Level Domain Key:\t" + urlz.getFirstLevelDomainKey());
        System.out.println("Second Level Domain Key:\t" + urlz.getSecondLevelDomainKey());
        System.out.println("Host:\t" + urlz.getHost());
        System.out.println("RawPage:\t" + urlz.getRawPage());
        System.out.println("Page:\t" + urlz.getPage());
        System.out.println("QueryParams:\t" + StringUtils.join(urlz.getQueryParams(),','));
        System.out.println("Folders:\t" + StringUtils.join(urlz.getFolders().toArray()));
        System.out.println("Public Suffix:\t" + urlz.getPublicSuffix());
    }


}